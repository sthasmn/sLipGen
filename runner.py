#!/usr/bin/env python3
"""
runner_full.py

Full end-to-end pipeline for sentence-level active speaker video clips:
 - Run LR-ASD (Columbia_test.py)
 - Detect FPS, load tracks & scores
 - Whisper transcription (audio from pyavi preferred)
 - Choose best track per segment (optional threshold)
 - Crop pyframes and mux with audio
 - Save clips and manifest.json

Supports multi-processing for large batch of videos.
"""

import sys, subprocess, json, math, shutil, re
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2
import pickle
import torch
import whisper
from multiprocessing import Pool, cpu_count, Lock, Manager

gpu_lock = Lock()
# ----------------- Utilities -----------------

def safe_load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

def detect_fps(base_dir: Path) -> float:
    for name in ("video.avi","video_only.avi","video_out.avi"):
        p = base_dir / "pyavi" / name
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0:
                return float(fps)
    return 25.0

def run_lr_asd(video_path: Path) -> Path:
    video_name = video_path.stem
    video_folder = str(video_path.parent)
    cmd = [sys.executable, "Columbia_test.py", "--videoName", video_name, "--videoFolder", video_folder]
    print(f"[LR-ASD] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return Path(video_folder) / video_name

def run_whisper_on_audio(audio_path: Path, model_size: Optional[str]=None, language: str="ja"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = model_size or ("large-v3-turbo" if device=="cuda" else "small")
    print(f"[Whisper] loading {model_size} on {device} ...")
    model = whisper.load_model(model_size, device=device)
    print(f"[Whisper] transcribing {audio_path} ...")
    res = model.transcribe(str(audio_path), language=language, fp16=(device=="cuda"), word_timestamps=True, verbose=False)
    segments = [{"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text","").strip()} for s in res.get("segments",[])]
    print(f"[Whisper] {len(segments)} segments")
    return segments

def load_tracks_scores(pywork_dir: Path):
    tracks_p = pywork_dir / "tracks.pckl"
    scores_p = pywork_dir / "scores.pckl"
    if not tracks_p.exists() or not scores_p.exists():
        raise FileNotFoundError(f"Missing tracks.pckl or scores.pckl in {pywork_dir}")
    tracks = safe_load_pickle(tracks_p)
    scores = safe_load_pickle(scores_p)
    scores_np = [np.asarray(s).flatten() for s in scores]
    return tracks, scores_np

def overlap_local_indices(track_frames: np.ndarray, start_f: int, end_f: int) -> np.ndarray:
    if track_frames.size == 0:
        return np.array([], dtype=int)
    return np.where((track_frames >= start_f) & (track_frames <= end_f))[0]

def choose_best_track(tracks, scores, fps: float, seg_start: float, seg_end: float, min_frames=2, smoothing=3, threshold: Optional[float]=None) -> Tuple[Optional[int], Optional[float]]:
    start_f = int(math.floor(seg_start * fps))
    end_f   = int(math.ceil(seg_end * fps))
    best_idx, best_val = None, -1e12
    for i, tr in enumerate(tracks):
        frame_arr = np.asarray(tr['track']['frame']).astype(int)
        if frame_arr.size == 0: continue
        score_arr = scores[i] if i < len(scores) else np.array([])
        if score_arr.size == 0: continue
        local_idxs = overlap_local_indices(frame_arr, start_f, end_f)
        if local_idxs.size < min_frames: continue
        local_idxs = local_idxs[local_idxs < score_arr.shape[0]]
        if local_idxs.size == 0: continue
        sel = score_arr[local_idxs]
        if smoothing > 1 and sel.size >= smoothing:
            sel = np.convolve(sel, np.ones(smoothing)/smoothing, mode='valid')
        val = float(np.mean(sel))
        if threshold is not None and val < threshold:
            continue
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx, (best_val if best_idx is not None else None)

def extract_cropped_video_from_pyframes(track_entry, pyframes_dir: Path, seg_start: float, seg_end: float,
                                        fps: float, out_video_path: Path, pad_mode='center') -> bool:
    track_frames = np.asarray(track_entry['track']['frame']).astype(int)
    bboxes = np.asarray(track_entry['track']['bbox']).astype(int)
    if track_frames.size == 0 or bboxes.size == 0: return False
    start_f = int(math.floor(seg_start * fps)); end_f = int(math.ceil(seg_end * fps))
    idxs = overlap_local_indices(track_frames, start_f, end_f)
    if idxs.size == 0: return False
    idxs = idxs[idxs < bboxes.shape[0]]
    widths = (bboxes[idxs,2] - bboxes[idxs,0]).clip(min=1)
    heights= (bboxes[idxs,3] - bboxes[idxs,1]).clip(min=1)
    target_w = int(min(1920, int(np.max(widths))))
    target_h = int(min(1080, int(np.max(heights))))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (target_w, target_h))
    if not writer.isOpened(): return False
    written = 0
    for idx in idxs:
        gf = int(track_frames[idx])
        frame_file = pyframes_dir / f"{gf:06d}.jpg"
        if not frame_file.exists():
            matches = list(pyframes_dir.glob(f"*{gf:03d}*.jpg"))
            frame_file = matches[0] if matches else None
            if frame_file is None: continue
        img = cv2.imread(str(frame_file))
        if img is None: continue
        h_img, w_img = img.shape[:2]
        x1,y1,x2,y2 = bboxes[idx]
        x1 = max(0,min(w_img-1,int(x1))); x2 = max(0,min(w_img,int(x2)))
        y1 = max(0,min(h_img-1,int(y1))); y2 = max(0,min(h_img,int(y2)))
        if x2<=x1 or y2<=y1: continue
        crop = img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        canvas = np.zeros((target_h,target_w,3),dtype=np.uint8)
        if pad_mode=='center':
            off_x = max(0,(target_w-cw)//2); off_y=max(0,(target_h-ch)//2)
        else:
            off_x=0; off_y=0
        ex = min(target_w, off_x+cw); ey = min(target_h, off_y+ch)
        canvas[off_y:ey, off_x:ex] = crop[0:(ey-off_y),0:(ex-off_x)]
        writer.write(canvas)
        written += 1
    writer.release()
    return written>0

def extract_audio_segment_from_pyavi(pyavi_audio: Path, seg_start: float, seg_end: float, out_audio: Path) -> bool:
    if not pyavi_audio.exists(): return False
    cmd = ["ffmpeg","-y","-i",str(pyavi_audio),"-ss",f"{seg_start:.3f}","-to",f"{seg_end:.3f}",
           "-vn","-acodec","aac","-b:a","128k",str(out_audio)]
    try: subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE); return True
    except subprocess.CalledProcessError: return False

def mux_audio_into_video(video_path: Path, audio_path: Path, out_path: Path) -> bool:
    if not video_path.exists() or not audio_path.exists(): return False
    cmd = ["ffmpeg","-y","-i",str(video_path),"-i",str(audio_path),"-c:v","copy","-c:a","aac","-shortest",str(out_path)]
    try: subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE); return True
    except subprocess.CalledProcessError: return False


def process_video(video_path: Path, threshold: Optional[float]=None):
    print(f"[start] {video_path}")
    try:
        base_out = run_lr_asd(video_path)
        pywork = base_out / "pywork"
        pyframes = base_out / "pyframes"
        pyavi_dir = base_out / "pyavi"
        output_dir = base_out / "output"; output_dir.mkdir(exist_ok=True)

        # Prefer pyavi audio
        pyavi_audio = pyavi_dir / "audio.wav"
        audio_for_transcription = pyavi_audio if pyavi_audio.exists() else video_path

        # ---- GPU-protected Whisper call ----
        # If a GPU lock was installed in the worker (via initializer), use it.
        global GPU_LOCK
        if 'GPU_LOCK' in globals() and GPU_LOCK is not None:
            with GPU_LOCK:
                segments = run_whisper_on_audio(audio_for_transcription, language="ja")
        else:
            segments = run_whisper_on_audio(audio_for_transcription, language="ja")
        # -------------------------------------

        tracks, scores = load_tracks_scores(pywork)
        fps = detect_fps(base_out)

        manifest = []
        for si, seg in enumerate(segments):
            sstart, send, text = float(seg["start"]), float(seg["end"]), seg["text"]
            best_idx, best_val = choose_best_track(tracks, scores, fps, sstart, send, threshold=threshold)
            if best_idx is None:
                manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": None, "clip": None})
                continue
            track_entry = tracks[best_idx]
            tmp_vid = output_dir / f"tmp_seg_{si:03d}_trk{best_idx+1}.mp4"
            final_vid = output_dir / f"seg_{si:03d}_trk{best_idx+1}.mp4"
            final_vid_audio = output_dir / f"seg_{si:03d}_trk{best_idx+1}_audio.mp4"

            ok_vid = extract_cropped_video_from_pyframes(track_entry, pyframes, sstart, send, fps, tmp_vid)
            if not ok_vid:
                manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": int(best_idx), "clip": None, "score": best_val})
                continue

            tmp_aac = output_dir / f"tmp_seg_{si:03d}.aac"
            audio_source = pyavi_audio if pyavi_audio.exists() else video_path
            ok_audio = extract_audio_segment_from_pyavi(audio_source, sstart, send, tmp_aac)
            if ok_audio:
                mux_audio_into_video(tmp_vid, tmp_aac, final_vid_audio)
                tmp_vid.unlink(missing_ok=True); tmp_aac.unlink(missing_ok=True)
                manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": int(best_idx), "clip": str(final_vid_audio), "score": float(best_val)})
            else:
                shutil.move(str(tmp_vid), str(final_vid))
                manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": int(best_idx), "clip": str(final_vid), "score": float(best_val)})

        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"[done] {video_path}")
    except Exception as e:
        print(f"[error] {video_path} -> {e}")

def _worker_init(gpu_lock_proxy):
    """
    Called once per worker process on startup. Installs a proxy lock as a global.
    """
    global GPU_LOCK
    GPU_LOCK = gpu_lock_proxy  # proxy Lock from Manager()

def main():
    if len(sys.argv)<2:
        print("Usage: python runner.py /path/to/videos_dir")
        sys.exit(1)
    videos_dir = Path(sys.argv[1])
    if not videos_dir.exists() or not videos_dir.is_dir():
        print("Invalid videos directory:", videos_dir)
        sys.exit(1)

    video_files = sorted(
        [f for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MXF", "*.mxf") for f in videos_dir.glob(ext)]
    )
    print(f"[info] {len(video_files)} videos found in {videos_dir}")

    threshold = 0.5  # optional: set minimum track score for speaker selection

    # Number of worker processes (CPU-bound tasks). Keep at most len(video_files).
    num_workers = min(cpu_count(), len(video_files))
    print(f"[info] using {num_workers} parallel workers")

    # Use a Manager() Lock so the lock object is proxy-picklable and visible in workers.
    manager = Manager()
    gpu_lock = manager.Lock()

    # Start pool with initializer that installs GPU_LOCK in each worker.
    with Pool(processes=num_workers, initializer=_worker_init, initargs=(gpu_lock,)) as p:
        # We intentionally pass the same process_video signature as before:
        p.starmap(process_video, [(vf, threshold) for vf in video_files])

    print("[all done] batch processing complete")



if __name__ == "__main__":
    main()