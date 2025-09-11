import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, json, math
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.detectors import ContentDetector
from model.faceDetector.s3fd import S3FD
from ASD import ASD
import python_speech_features

# This script is a direct modification of the original, working Columbia_test.py
# Its sole purpose is to run the proven ASD pipeline and output a JSON file
# with the final, smoothed bounding boxes and scores.

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Definitive ASD Script")
parser.add_argument('--videoName', type=str, required=True)
parser.add_argument('--videoFolder', type=str, required=True)
parser.add_argument('--outputJsonPath', type=str, required=True)
parser.add_argument('--pretrainModel', type=str, default="weight/finetuning_TalkSet.model")
parser.add_argument('--score_threshold', type=float, default=0.0)
# Arguments from original script, kept for consistency
parser.add_argument('--nDataLoaderThread', type=int, default=10)
parser.add_argument('--facedetScale', type=float, default=0.25)
parser.add_argument('--minTrack', type=int, default=10)
parser.add_argument('--numFailedDet', type=int, default=10)
parser.add_argument('--minFaceSize', type=int, default=1)
parser.add_argument('--cropScale', type=float, default=0.40)

args = parser.parse_args()


def scene_detect(args):
    videoManager = VideoManager([args.videoFilePath]);
    sceneManager = SceneManager()
    sceneManager.add_detector(ContentDetector());
    videoManager.start();
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list()
    if not sceneList:
        try:
            cap = cv2.VideoCapture(args.videoFilePath);
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
            cap.release()
            sceneList = [(FrameTimecode(timecode='00:00:00.000', fps=25.0), FrameTimecode(frame=frameCount, fps=25.0))]
        except Exception:
            sceneList = []
    return sceneList


def inference_video(args):
    DET = S3FD(device='cuda');
    flist = sorted(glob.glob(os.path.join(args.pyframesPath, '*.jpg')));
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname);
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale]);
        dets.append([])
        for bbox in bboxes: dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist()})
        sys.stderr.write(f'{os.path.basename(args.videoFilePath)}-{fidx:05d}; {len(dets[-1])} dets\r')
    return dets


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0]);
    yA = max(boxA[1], boxB[1]);
    xB = min(boxA[2], boxB[2]);
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def track_shot(args, sceneFaces):
    iouThres, tracks = 0.5, []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if not track:
                    track.append(face); frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet and bb_intersection_over_union(
                        face['bbox'], track[-1]['bbox']) > iouThres:
                    track.append(face);
                    frameFaces.remove(face)
        if not track:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f['frame'] for f in track]);
            bboxes = numpy.array([f['bbox'] for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1);
            bboxesI = []
            for ij in range(4): bboxesI.append(
                interp1d(frameNum, bboxes[:, ij], kind='linear', fill_value="extrapolate")(frameI))
            tracks.append({'frame': frameI, 'bbox': numpy.stack(bboxesI, axis=1)})
    return tracks


def crop_video(args, track, cropFile):
    flist = sorted(glob.glob(os.path.join(args.pyframesPath, '*.jpg')))
    fourcc = cv2.VideoWriter_fourcc(*'XVID');
    vOut = cv2.VideoWriter(cropFile + 't.avi', fourcc, 25, (224, 224))
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2);
        dets['y'].append((det[1] + det[3]) / 2);
        dets['x'].append((det[0] + det[2]) / 2)
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13);
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13);
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        bsi = int(dets['s'][fidx] * (1 + 2 * args.cropScale));
        image = cv2.imread(flist[frame])
        frame_pad = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi;
        mx = dets['x'][fidx] + bsi
        face = frame_pad[int(my - dets['s'][fidx]):int(my + dets['s'][fidx] * (1 + 2 * args.cropScale)),
               int(mx - dets['s'][fidx] * (1 + args.cropScale)):int(mx + dets['s'][fidx] * (1 + args.cropScale))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp, audioStart, audioEnd = cropFile + '.wav', (track['frame'][0]) / 25, (track['frame'][-1] + 1) / 25
    vOut.release()
    command = (
        f"ffmpeg -y -i {args.audioFilePath} -ac 1 -vn -acodec pcm_s16le -ar 16000 -ss {audioStart} -to {audioEnd} {audioTmp} -loglevel quiet")
    subprocess.call(command, shell=True)
    command = (f"ffmpeg -y -i {cropFile}t.avi -i {audioTmp} -c:v copy -c:a copy {cropFile}.avi -loglevel quiet")
    subprocess.call(command, shell=True)
    os.remove(cropFile + 't.avi')
    return {'track': track, 'proc_track': dets}


def evaluate_network(files, args):
    s = ASD();
    s.loadParameters(args.pretrainModel);
    sys.stderr.write(f"Model {args.pretrainModel} loaded! \r\n");
    s.eval()
    allScores, durationSet = [], {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
    for file in tqdm.tqdm(files, total=len(files), desc="Scoring tracks"):
        fileName = os.path.splitext(os.path.basename(file))[0]
        try:
            _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
            audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        except Exception:
            allScores.append(numpy.array([])); continue
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY);
                face = cv2.resize(face, (224, 224))
                face = face[int(112 - 56):int(112 + 56), int(112 - 56):int(112 + 56)];
                videoFeature.append(face)
            else:
                break
        video.release();
        videoFeature = numpy.array(videoFeature)
        if videoFeature.ndim < 2: allScores.append(numpy.array([])); continue
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / 25)
        audioFeature = audioFeature[:int(round(length * 100)), :];
        videoFeature = videoFeature[:int(round(length * 25)), :, :]
        allScore = []
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration));
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(
                        0).cuda()
                    inputV = torch.FloatTensor(videoFeature[i * duration * 25:(i + 1) * duration * 25, :, :]).unsqueeze(
                        0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA);
                    embedV = s.model.forward_visual_frontend(inputV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV);
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        allScores.append(numpy.round((numpy.mean(allScore, axis=0)), 1).astype(float))
    return allScores


def save_output_to_json(vidTracks, scores, args):
    """This function uses the proven logic from the original script to save the final JSON."""
    results = []
    faces_by_frame = [[] for _ in range(args.total_frames)]
    for tidx, track in enumerate(vidTracks):
        # The 'vidTracks' items are dicts: {'track':..., 'proc_track':...}
        if tidx >= len(scores) or not scores[tidx].any(): continue
        score = scores[tidx]
        for fidx, frame_num in enumerate(track['track']['frame'].tolist()):
            s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)])
            if frame_num < len(faces_by_frame):
                is_active = bool(s >= args.score_threshold)
                # Re-calculate the bbox from the smoothed, processed track data (`proc_track`)
                # This is the "secret sauce" that creates the clean green box
                x = track['proc_track']['x'][fidx]
                y = track['proc_track']['y'][fidx]
                sz = track['proc_track']['s'][fidx]
                bbox = [int(x - sz), int(y - sz), int(x + sz), int(y + sz)]
                faces_by_frame[frame_num].append({
                    'track_id': tidx,
                    'score': float(s),
                    'is_active': is_active,
                    'bbox': bbox
                })

    for frame_num, faces in enumerate(faces_by_frame):
        if faces: results.append({'frame': frame_num, 'faces': faces})

    with open(args.outputJsonPath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nASD results with proven smoothed bounding boxes saved to: {args.outputJsonPath}")


def main():
    args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
    folder_name = 'asd_temp'
    args.savePath = os.path.join(args.videoFolder, args.videoName, folder_name)

    args.pyaviPath = os.path.join(args.savePath, 'pyavi');
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork');
    args.pycropPath = os.path.join(args.savePath, 'pycrop')

    if os.path.exists(args.savePath): rmtree(args.savePath)
    os.makedirs(args.pyaviPath, exist_ok=True);
    os.makedirs(args.pyframesPath, exist_ok=True)
    os.makedirs(args.pyworkPath, exist_ok=True);
    os.makedirs(args.pycropPath, exist_ok=True)

    print("--- Running Definitive ASD Script ---")
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    command = (f'ffmpeg -y -i "{args.videoPath}" -qscale:v 2 -r 25 -async 1 "{args.videoFilePath}" -loglevel quiet')
    subprocess.call(command, shell=True)
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = (f'ffmpeg -y -i "{args.videoFilePath}" -ac 1 -vn -ar 16000 "{args.audioFilePath}" -loglevel quiet')
    subprocess.call(command, shell=True)
    command = (
        f'ffmpeg -y -i "{args.videoFilePath}" -f image2 "{os.path.join(args.pyframesPath, "%06d.jpg")}" -loglevel quiet')
    subprocess.call(command, shell=True)
    print("Step 1: Frames and audio extracted.")
    args.total_frames = len(glob.glob(os.path.join(args.pyframesPath, '*.jpg')))

    scene = scene_detect(args);
    print(f"Step 2: Found {len(scene)} scenes.")
    faces = inference_video(args);
    print(f"\nStep 3: Face detection complete.")
    allTracks = track_shot(args, faces);
    print(f"Step 4: Found {len(allTracks)} potential tracks.")
    if not allTracks: print("No face tracks found. Exiting."); return

    vidTracks = []
    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks), desc="Cropping intermediate clips"):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d' % ii)))

    files = glob.glob(f"{args.pycropPath}/*.avi");
    files.sort()
    scores = evaluate_network(files, args);
    print("Step 5: Scoring complete.")

    save_output_to_json(vidTracks, scores, args)

    rmtree(args.savePath)
    print(f"--- ASD Script Finished ---")


if __name__ == '__main__':
    main()

