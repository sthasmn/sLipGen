import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, json, python_speech_features
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from model.faceDetector.s3fd import S3FD
from ASD import ASD

warnings.filterwarnings("ignore")

# --- Original parser from the script ---
parser = argparse.ArgumentParser(description="Columbia ASD Evaluation")
parser.add_argument('--videoName', type=str, default="col", help='Demo video name')
parser.add_argument('--videoFolder', type=str, default="colDataPath", help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel', type=str, default="weight/finetuning_TalkSet.model",
                    help='Path for the pretrained model')
parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of workers')
parser.add_argument('--facedetScale', type=float, default=0.25,
                    help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack', type=int, default=10, help='Number of min frames for each shot')
parser.add_argument('--numFailedDet', type=int, default=10,
                    help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize', type=int, default=1, help='Minimum face size in pixels')
parser.add_argument('--cropScale', type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--start', type=int, default=0, help='The start time of the video')
parser.add_argument('--duration', type=int, default=0,
                    help='The duration of the video, when set as 0, will extract the whole video')
parser.add_argument('--evalCol', dest='evalCol', action='store_true', help='Evaluate on Columbia dataset')
parser.add_argument('--colSavePath', type=str, default="/colDataPath", help='Path for inputs, tmps and outputs')

# --- NEW ARGUMENT FOR OUR PIPELINE ---
parser.add_argument('--outputJsonPath', type=str, default='asd_results.json',
                    help='Path to save the ASD results in JSON format')

args = parser.parse_args()

# The original script has logic for downloading data, we remove it for our use case.
args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
args.savePath = os.path.join(args.videoFolder, args.videoName, "asd_temp")  # Use a subfolder for temp files


# --- All helper functions from the original script are kept the same ---
# scene_detect, inference_video, bb_intersection_over_union, track_shot,
# crop_video, extract_MFCC, evaluate_network are all here, unchanged.

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
    savePath = os.path.join(args.pyworkPath, 'scene.pckl')
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(), videoManager.get_current_timecode())]
    with open(savePath, 'wb') as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write('%s - scenes detected %d\n' % (args.videoFilePath, len(sceneList)))
    return sceneList


def inference_video(args):
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
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
    savePath = os.path.join(args.pyworkPath, 'faces.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(dets, fil)
    return dets


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(args, sceneFaces):
    iouThres = 0.5
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if max(numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                   numpy.mean(bboxesI[:, 3] - bboxesI[:, 1])) > args.minFaceSize:
                tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks


def crop_video(args, track, cropFile):
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224, 224))
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = args.cropScale
        bs = dets['s'][fidx]
        bsi = int(bs * (1 + 2 * cs))
        image = cv2.imread(flist[frame])
        frame_pad = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi
        mx = dets['x'][fidx] + bsi
        face = frame_pad[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + '.wav'
    audioStart = (track['frame'][0]) / 25
    audioEnd = (track['frame'][-1] + 1) / 25
    vOut.release()
    command = (
                "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
                (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp))
    output = subprocess.call(command, shell=True, stdout=None)
    _, audio = wavfile.read(audioTmp)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
               (cropFile, audioTmp, args.nDataLoaderThread, cropFile))
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track': track, 'proc_track': dets}


def evaluate_network(files, args):
    s = ASD()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n" % args.pretrainModel)
    s.eval()
    allScores = []
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split('/')[-1])[0]
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112 - (112 / 2)):int(112 + (112 / 2)), int(112 - (112 / 2)):int(112 + (112 / 2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * 25)), :, :]
        allScore = []
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i + 1) * duration * 100, :]).unsqueeze(
                        0).cuda()
                    inputV = torch.FloatTensor(
                        videoFeature[i * duration * 25: (i + 1) * duration * 25, :, :]).unsqueeze(0).cuda()
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
        allScores.append(allScore)
    return allScores


# --- NEW FUNCTION TO SAVE RESULTS TO JSON ---
def save_results_to_json(tracks, scores, args):
    """
    This function replaces the 'visualization' function.
    It processes the tracks and scores and saves them to a structured JSON file.
    """
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()

    # Create a dictionary to hold results for each frame
    frame_results = {}

    for tidx, track in enumerate(tracks):
        track_scores = scores[tidx]
        for fidx, frame_index in enumerate(track['track']['frame'].tolist()):
            # Perform score smoothing like in the original visualization
            s = track_scores[max(fidx - 2, 0): min(fidx + 3, len(track_scores) - 1)]
            score = numpy.mean(s)

            if frame_index not in frame_results:
                frame_results[frame_index] = {'frame_index': frame_index, 'faces': []}

            proc_track = track['proc_track']
            center_x = proc_track['x'][fidx]
            center_y = proc_track['y'][fidx]
            size = proc_track['s'][fidx]

            face_data = {
                'box': [
                    int(center_x - size),
                    int(center_y - size),
                    int(center_x + size),
                    int(center_y + size)
                ],
                'score': float(score),
                'active': bool(score >= 0)  # Simple threshold: positive score means active
            }
            frame_results[frame_index]['faces'].append(face_data)

    # Convert dictionary to a sorted list of frames
    sorted_frames = sorted(frame_results.values(), key=lambda x: x['frame_index'])

    # Get video FPS
    cap = cv2.VideoCapture(args.videoFilePath)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    final_output = {
        'video_fps': video_fps,
        'frames': sorted_frames
    }

    with open(args.outputJsonPath, 'w') as f:
        json.dump(final_output, f, indent=4)

    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" ASD results saved to {args.outputJsonPath} \r\n")


def main():
    # Initialization
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.pyaviPath, exist_ok=True)
    os.makedirs(args.pyframesPath, exist_ok=True)
    os.makedirs(args.pyworkPath, exist_ok=True)
    os.makedirs(args.pycropPath, exist_ok=True)

    # Extract video
    args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
    if args.duration == 0:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
                   (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
    else:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
                   (args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" % (args.videoFilePath))

    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
               (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" % (args.audioFilePath))

    # Extract the video frames
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
               (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg')))
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" % (args.pyframesPath))

    # Scene detection for the video frames
    scene = scene_detect(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" % (args.pyworkPath))

    # Face detection for the video frames
    faces = inference_video(args)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" % (args.pyworkPath))

    # Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num]))
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" % len(allTracks))

    # Face clips cropping
    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d' % ii)))
    savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
    with open(savePath, 'wb') as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" % args.pycropPath)

    # Active Speaker Detection
    files = glob.glob("%s/*.avi" % args.pycropPath)
    files.sort()
    scores = evaluate_network(files, args)

    # --- MODIFIED PART ---
    # Instead of visualizing, we save the results to a JSON file.
    save_results_to_json(vidTracks, scores, args)

    # Clean up temporary files created by the ASD script
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Cleaning up temporary files... \r\n")
    rmtree(args.savePath)


if __name__ == '__main__':
    main()
