import sys
import os
import subprocess
import json
import math
import shutil
from pathlib import Path
import cv2
import torch
import whisper
import numpy as np
import pickle
import glob
import time
import tqdm
from typing import Optional, List, Dict, Any

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import python_speech_features

from model.faceDetector.s3fd import S3FD
from ASD import ASD


# A simple data class to pass between pipeline stages
class PipelineTask:
    def __init__(self, video_path: Path, base_dir: Path):
        self.video_path = video_path
        self.base_dir = base_dir
        self.pyavi_path = base_dir / 'pyavi'
        self.pyframes_path = base_dir / 'pyframes'
        self.pywork_path = base_dir / 'pywork'
        self.pycrop_path = base_dir / 'pycrop'
        self.output_path = base_dir / 'output'
        self.audio_path = self.pyavi_path / 'audio.wav'
        self.fps = 25.0
        self.tracks = None
        self.scores = None
        self.transcription = None


# =====================================================================================
#  Core Logic Functions (ported from integrated_runner.py)
# =====================================================================================

def scene_detect(video_path: Path, work_dir: Path):
    video_manager = VideoManager([str(video_path)])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.release()
        scene_list = [(0.0, frame_count / fps)]
    else:
        scene_list = [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]

    save_path = work_dir / 'scene.pckl'
    with open(save_path, 'wb') as fil:
        pickle.dump(scene_list, fil)
    sys.stderr.write(f'{video_path.name} - scenes detected {len(scene_list)}\n')
    return scene_list


def inference_video(frames_dir: Path, work_dir: Path, config):
    det = S3FD(device=config.VISION_DEVICE)
    flist = sorted(glob.glob(str(frames_dir / '*.jpg')))
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        if image is None: continue
        image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = det.detect_faces(image_numpy, conf_th=0.9, scales=[config.FACEDET_SCALE])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
        sys.stderr.write(
            f'{frames_dir.parent.name} - Face detection frame {fidx + 1}/{len(flist)}; {len(dets[-1])} dets\r')

    save_path = work_dir / 'faces.pckl'
    with open(save_path, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou


def track_shot(scene_faces, config):
    iou_thres = 0.5
    tracks = []
    while True:
        track = []
        for frame_faces in scene_faces:
            if frame_faces:
                track.append(frame_faces.pop(0))
                break
        if not track:
            break

        for frame_faces in scene_faces:
            for face in frame_faces:
                if face['frame'] - track[-1]['frame'] <= config.NUM_FAILED_DET:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iou_thres:
                        track.append(face)
                        frame_faces.remove(face)
                        break

        if len(track) > config.MIN_TRACK:
            frame_num = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])
            frame_i = np.arange(frame_num[0], frame_num[-1] + 1)
            bboxes_i = []
            for ij in range(4):
                interpfn = interp1d(frame_num, bboxes[:, ij], kind='linear', bounds_error=False,
                                    fill_value="extrapolate")
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                   np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > config.MIN_FACE_SIZE:
                tracks.append({'frame': frame_i, 'bbox': bboxes_i})
    return tracks


def crop_video(track, frames_dir: Path, audio_path: Path, crop_file_path: Path, config, fps: float):
    flist = sorted(glob.glob(str(frames_dir / '*.jpg')))
    vOut = cv2.VideoWriter(str(crop_file_path) + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (224, 224))

    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)

    kernel_size = min(13, len(dets['s']) if len(dets['s']) % 2 != 0 else len(dets['s']) - 1)
    if kernel_size > 0:
        dets['s'] = signal.medfilt(dets['s'], kernel_size=kernel_size)
        dets['x'] = signal.medfilt(dets['x'], kernel_size=kernel_size)
        dets['y'] = signal.medfilt(dets['y'], kernel_size=kernel_size)

    for fidx, frame_num in enumerate(track['frame']):
        frame_index = int(frame_num)
        if frame_index >= len(flist): continue

        image = cv2.imread(flist[frame_index])
        if image is None: continue

        cs = config.CROP_SCALE
        bs = dets['s'][fidx]
        bsi = int(bs * (1 + 2 * cs))

        frame_padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi
        mx = dets['x'][fidx] + bsi

        y1, y2 = int(my - bs), int(my + bs * (1 + 2 * cs))
        x1, x2 = int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs))

        face = frame_padded[y1:y2, x1:x2]
        vOut.write(cv2.resize(face, (224, 224)))

    vOut.release()

    audio_tmp = str(crop_file_path) + '.wav'
    audio_start = track['frame'][0] / fps
    audio_end = (track['frame'][-1] + 1) / fps

    command = f"ffmpeg -y -i {audio_path} -ss {audio_start:.3f} -to {audio_end:.3f} -ac 1 -ar 16000 {audio_tmp} -loglevel panic"
    subprocess.call(command, shell=True)

    if not os.path.exists(audio_tmp) or os.path.getsize(audio_tmp) == 0:
        if os.path.exists(str(crop_file_path) + 't.avi'): os.remove(str(crop_file_path) + 't.avi')
        return None

    command = f"ffmpeg -y -i {crop_file_path}t.avi -i {audio_tmp} -c:v copy -c:a copy {crop_file_path}.avi -loglevel panic"
    subprocess.call(command, shell=True)

    if os.path.exists(str(crop_file_path) + 't.avi'): os.remove(str(crop_file_path) + 't.avi')
    return {'track': track, 'proc_track': dets}


def evaluate_network(pycrop_path: Path, asd_model: ASD, config):
    files = sorted(glob.glob(str(pycrop_path / '*.avi')))
    all_scores = []

    for file in tqdm.tqdm(files, desc="[ASD Scoring]", leave=False):
        try:
            _, audio = wavfile.read(file.replace('.avi', '.wav'))
            if len(audio) < 1600:
                all_scores.append([])
                continue

            audio_feature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

            video = cv2.VideoCapture(file)
            video_feature = []
            while video.isOpened():
                ret, frames = video.read()
                if not ret: break
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[56:168, 56:168]
                video_feature.append(face)
            video.release()
            video_feature = np.array(video_feature)

            length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) // 100, video_feature.shape[0] // 25)
            if length <= 0:
                all_scores.append([])
                continue

            audio_feature = audio_feature[:int(round(length * 100)), :]
            video_feature = video_feature[:int(round(length * 25)), :, :]

            scores = []
            with torch.no_grad():
                input_a = torch.FloatTensor(audio_feature).unsqueeze(0).to(config.VISION_DEVICE)
                input_v = torch.FloatTensor(video_feature).unsqueeze(0).to(config.VISION_DEVICE)
                embed_a = asd_model.model.forward_audio_frontend(input_a)
                embed_v = asd_model.model.forward_visual_frontend(input_v)
                out = asd_model.model.forward_audio_visual_backend(embed_a, embed_v)
                score = asd_model.lossAV.forward(out, labels=None)
                scores.extend(score)

            all_scores.append(scores)
        except Exception as e:
            print(f"Warning: Could not score {file}. Reason: {e}")
            all_scores.append([])

    return all_scores


def create_final_clip(track_data, start_time, end_time, frames_dir, audio_path, output_path, config, fps):
    tmp_video_path = output_path.with_suffix('.tmp.mp4')
    tmp_audio_path = output_path.with_suffix('.tmp.aac')

    all_frame_files = sorted(glob.glob(str(frames_dir / '*.jpg')))
    writer = cv2.VideoWriter(str(tmp_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (224, 224))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    track_frames_map = {int(f): i for i, f in enumerate(track_data['track']['frame'])}

    for frame_num in range(start_frame, end_frame + 1):
        if frame_num in track_frames_map and frame_num < len(all_frame_files):
            track_idx = track_frames_map[frame_num]
            image = cv2.imread(all_frame_files[frame_num])
            if image is None: continue

            proc_track = track_data['proc_track']
            cs = config.CROP_SCALE
            bs = proc_track['s'][track_idx]
            bsi = int(bs * (1 + 2 * cs))

            frame_padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my = proc_track['y'][track_idx] + bsi
            mx = proc_track['x'][track_idx] + bsi

            y1, y2 = int(my - bs), int(my + bs * (1 + 2 * cs))
            x1, x2 = int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs))

            face = frame_padded[y1:y2, x1:x2]
            writer.write(cv2.resize(face, (224, 224)))
    writer.release()

    cmd_audio = f"ffmpeg -y -i {audio_path} -ss {start_time:.3f} -to {end_time:.3f} -c:a aac {tmp_audio_path} -loglevel panic"
    subprocess.run(cmd_audio, shell=True, check=True)

    cmd_mux = f"ffmpeg -y -i {tmp_video_path} -i {tmp_audio_path} -c:v libx264 -c:a copy {output_path} -loglevel panic"
    subprocess.run(cmd_mux, shell=True, check=True)

    os.remove(tmp_video_path)
    os.remove(tmp_audio_path)


# =====================================================================================
#  Pipeline Stage Functions
# =====================================================================================

def run_video_preprocessing(video_path: Path, config) -> Optional[PipelineTask]:
    base_dir = video_path.parent / video_path.stem
    task = PipelineTask(video_path, base_dir)

    for p in [task.pyavi_path, task.pyframes_path, task.pywork_path, task.pycrop_path, task.output_path]:
        os.makedirs(p, exist_ok=True)

    subprocess.run(['ffmpeg', '-y', '-i', str(video_path), '-qscale:a', '0', '-ac', '1', '-vn', '-ar', '16000',
                    str(task.audio_path), '-loglevel', 'panic'], check=True)
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(video_path), '-qscale:v', '2', '-f', 'image2', str(task.pyframes_path / '%06d.jpg'),
         '-loglevel', 'panic'], check=True)

    cap = cv2.VideoCapture(str(video_path))
    task.fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    return task


def run_vision_pipeline(task: PipelineTask, config) -> PipelineTask:
    scenes = scene_detect(task.video_path, task.pywork_path)
    faces = inference_video(task.pyframes_path, task.pywork_path, config)

    all_tracks = []
    for start_sec, end_sec in scenes:
        start_frame = int(start_sec * task.fps)
        end_frame = int(end_sec * task.fps)

        face_groups = [[] for _ in range(end_frame - start_frame)]
        for f_list in faces[start_frame:end_frame]:
            for face in f_list:
                local_frame_idx = face['frame'] - start_frame
                if 0 <= local_frame_idx < len(face_groups):
                    face_groups[local_frame_idx].append(face)

        all_tracks.extend(track_shot(face_groups, config))
    print(f"\n[Tracker] Found {len(all_tracks)} tracks for {task.video_path.name}.")

    vid_tracks = []
    for i, track in enumerate(tqdm.tqdm(all_tracks, desc="[Cropping]", leave=False)):
        cropped = crop_video(track, task.pyframes_path, task.audio_path, task.pycrop_path / f'{i:05d}', config,
                             task.fps)
        if cropped:
            vid_tracks.append(cropped)

    with open(task.pywork_path / 'tracks.pckl', 'wb') as f:
        pickle.dump(vid_tracks, f)

    asd_model = ASD().to(config.VISION_DEVICE)
    asd_model.loadParameters(config.PRETRAIN_MODEL_PATH)
    asd_model.eval()

    scores = evaluate_network(task.pycrop_path, asd_model, config)

    with open(task.pywork_path / 'scores.pckl', 'wb') as f:
        pickle.dump(scores, f)

    task.tracks = vid_tracks
    task.scores = scores
    return task


def run_transcription(task: PipelineTask, model, config) -> PipelineTask:
    task.transcription = model.transcribe(
        str(task.audio_path),
        language=config.WHISPER_LANGUAGE,
        fp16=(config.WHISPER_DEVICE != "cpu"),
        word_timestamps=True,
        verbose=False
    )
    return task


def run_final_muxing(task: PipelineTask, config):
    manifest = []
    for seg_idx, segment in enumerate(task.transcription["segments"]):
        text = segment["text"].strip()
        if not text: continue

        start_time = segment["start"]
        end_time = segment["end"]

        best_track_idx, best_avg_score = -1, -1.0

        for track_idx, track_scores in enumerate(task.scores):
            if not track_scores or track_idx >= len(task.tracks): continue

            track_start_frame = task.tracks[track_idx]['track']['frame'][0]
            track_end_frame = task.tracks[track_idx]['track']['frame'][-1]

            seg_start_frame = int(start_time * task.fps)
            seg_end_frame = int(end_time * task.fps)

            if seg_end_frame < track_start_frame or seg_start_frame > track_end_frame:
                continue

            avg_score = np.mean(track_scores)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_track_idx = track_idx

        if best_track_idx != -1 and best_avg_score > config.MIN_SPEAKER_SCORE_THRESHOLD:
            clip_filename = f"seg_{seg_idx:04d}_track_{best_track_idx:03d}.mp4"
            clip_path = task.output_path / clip_filename

            create_final_clip(
                track_data=task.tracks[best_track_idx],
                start_time=start_time,
                end_time=end_time,
                frames_dir=task.pyframes_path,
                audio_path=task.audio_path,
                output_path=clip_path,
                config=config,
                fps=task.fps
            )

            manifest.append({
                "clip": clip_filename,
                "text": text,
                "start_time": start_time,
                "end_time": end_time,
                "speaker_track": best_track_idx,
                "avg_score": float(best_avg_score)
            })

    with open(task.output_path / "manifest.json", "w", encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

