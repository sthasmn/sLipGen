import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, json
import python_speech_features
from shutil import rmtree
import pprint

from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d

from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from ASD import ASD

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Visual Debugger for Active Speaker Detection")

# --- Command line arguments ---
parser.add_argument('--videoName', type=str, help='Demo video name')
parser.add_argument('--videoFolder', type=str, help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel', type=str, default="weight/pretrain_AVA.model",
                    help='Path for the pretrained model')
parser.add_argument('--facedetScale', type=float, default=0.25, help='Scale factor for face detection')
parser.add_argument('--minTrack', type=int, default=10, help='Min frames for each shot')

# --- Other settings (usually no need to change) ---
parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of workers')
parser.add_argument('--numFailedDet', type=int, default=10, help='Number of missed detections allowed')
parser.add_argument('--minFaceSize', type=int, default=1, help='Minimum face size in pixels')
parser.add_argument('--cropScale', type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--start', type=int, default=0, help='The start time of the video')
parser.add_argument('--duration', type=int, default=0, help='The duration of the video')

args = parser.parse_args()


def scene_detect(args):
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    if not sceneList:
        sceneList = [(videoManager.get_base_timecode(), videoManager.get_current_timecode())]
    return sceneList


def inference_video(args):
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'));
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
        sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
    return dets


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0]);
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(args, sceneFaces):
    iouThres = 0.5;
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if not track:
                    track.append(face); frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres: track.append(face); frameFaces.remove(face); continue
                else:
                    break
        if not track:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4): bboxesI.append(interp1d(frameNum, bboxes[:, ij])(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                   numpy.mean(bboxesI[:, 3] - bboxesI[:, 1])) > args.minFaceSize:
                tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks


def crop_video(args, track, cropFile):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'));
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2);
        dets['x'].append((det[0] + det[2]) / 2)
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13);
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13);
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = args.cropScale;
        bs = dets['s'][fidx];
        bsi = int(bs * (1 + 2 * cs))
        image = cv2.imread(flist[frame])
        frame_pad = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi;
        mx = dets['x'][fidx] + bsi
        face = frame_pad[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + '.wav';
    audioStart = (track['frame'][0]) / 25;
    audioEnd = (track['frame'][-1] + 1) / 25
    vOut.release()
    command = (
                "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
                (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
    subprocess.call(command, shell=True, stdout=None)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
               (cropFile, audioTmp, args.nDataLoaderThread, cropFile))
    subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track': track, 'proc_track': dets}


def evaluate_network(files, args):
    s = ASD();
    s.loadParameters(args.pretrainModel);
    sys.stderr.write("Model %s loaded! \r\n" % args.pretrainModel);
    s.eval()
    allScores = [];
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
    for file in tqdm.tqdm(files, total=len(files), desc="Scoring tracks"):
        fileName = os.path.splitext(file.split('/')[-1])[0]
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY);
                face = cv2.resize(face, (224, 224))
                face = face[int(112 - (112 / 2)):int(112 + (112 / 2)), int(112 - (112 / 2)):int(112 + (112 / 2))]
                videoFeature.append(face)
            else:
                break
        video.release();
        videoFeature = numpy.array(videoFeature)
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
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores


def visualization(tracks, scores, args):
    # --- ADDED FOR DEBUGGING ---
    # This section will print the raw scores to the console for inspection.
    print("\n\n--- Raw Score Analysis ---")
    print(f"Number of tracks detected: {len(scores)}")
    print("Scores per track (each line is a list of scores for a tracked face):")
    pprint.pprint(scores)
    print("--------------------------\n")
    # --- END OF DEBUGGING ADDITION ---

    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'));
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)])
            faces[frame].append({'track': tidx, 'score': float(s), 's': track['proc_track']['s'][fidx],
                                 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})

    output_video_path = os.path.join(args.savePath, 'video_out.avi')
    firstImage = cv2.imread(flist[0]);
    fw = firstImage.shape[1];
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (fw, fh))
    colorDict = {0: (0, 0, 255), 1: (0, 255, 0)}

    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist), desc="Rendering video"):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            is_active = int(face['score'] >= 0);
            clr = colorDict[is_active]
            x1, y1, x2, y2 = int(face['x'] - face['s']), int(face['y'] - face['s']), int(face['x'] + face['s']), int(
                face['y'] + face['s'])
            cv2.rectangle(image, (x1, y1), (x2, y2), clr, 3)
            score_text = f"{face['score']:.1f}"
            cv2.putText(image, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 2)
        vOut.write(image)
    vOut.release()
    print(f"\nDebug video saved to: {output_video_path}")


def main():
    args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
    args.savePath = os.path.join(args.videoFolder, args.videoName, 'asd_temp_DEBUG')

    args.pyaviPath = os.path.join(args.savePath, 'pyavi');
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork');
    args.pycropPath = os.path.join(args.savePath, 'pycrop')

    if os.path.exists(args.savePath): rmtree(args.savePath)
    os.makedirs(args.pyaviPath, exist_ok=True);
    os.makedirs(args.pyframesPath, exist_ok=True)
    os.makedirs(args.pyworkPath, exist_ok=True);
    os.makedirs(args.pycropPath, exist_ok=True)

    print("--- Starting ASD Visual Debugger ---")
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    command = (
        f'ffmpeg -y -i "{args.videoPath}" -qscale:v 2 -threads {args.nDataLoaderThread} -async 1 -r 25 "{args.videoFilePath}" -loglevel quiet')
    subprocess.call(command, shell=True, stdout=None)

    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = (f'ffmpeg -y -i "{args.videoFilePath}" -ac 1 -vn -ar 16000 "{args.audioFilePath}" -loglevel quiet')
    subprocess.call(command, shell=True, stdout=None)

    command = (
        f'ffmpeg -y -i "{args.videoFilePath}" -f image2 "{os.path.join(args.pyframesPath, "%06d.jpg")}" -loglevel quiet')
    subprocess.call(command, shell=True, stdout=None)
    print("Step 1: Video, audio, and frames extracted.")

    scene = scene_detect(args);
    print(f"Step 2: Scene detection complete. Found {len(scene)} scenes.")
    faces = inference_video(args);
    print(f"\nStep 3: Face detection complete.")
    allTracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num]))
    print(f"Step 4: Face tracking complete. Found {len(allTracks)} potential tracks.")
    if not allTracks: print("\nWARNING: No face tracks were found."); return

    vidTracks = []
    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks), desc="Cropping faces"):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d' % ii)))
    print("Step 5: Face cropping complete.")

    files = glob.glob(f"{args.pycropPath}/*.avi");
    files.sort()
    scores = evaluate_network(files, args);
    print("Step 6: Active Speaker Detection scoring complete.")
    visualization(vidTracks, scores, args)

    print(f"\n--- Debugging Complete ---")
    print(f"Intermediate files are saved in: {args.savePath}")


if __name__ == '__main__':
    main()
