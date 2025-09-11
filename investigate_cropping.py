import sys, os, glob, subprocess, warnings, cv2, pickle, numpy
from scipy import signal
from scipy.interpolate import interp1d
from model.faceDetector.s3fd import S3FD
import argparse
from pathlib import Path

# --- This script is designed to investigate the bounding box smoothing logic ---

# Suppress warnings
warnings.filterwarnings("ignore")


# --- Helper functions copied directly from the original Columbia_test.py ---

def inference_video(args, pyframesPath):
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(pyframesPath, '*.jpg'));
    flist.sort()
    dets = []
    print("Running face detection on a few frames for analysis...")
    # Process enough frames to get a stable track
    for fidx, fname in enumerate(tqdm(flist[:100], desc="Detecting faces")):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[0.25])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({'frame': fidx, 'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
    return dets


def track_shot(faces):
    iouThres, tracks, numFailedDet, minTrack = 0.5, [], 10, 10
    while True:
        track = []
        for frameFaces in faces:
            for face in frameFaces:
                if not track:
                    track.append(face); frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres: track.append(face); frameFaces.remove(face); continue
                else:
                    break
        if not track:
            break
        elif len(track) > minTrack:
            frameNum = numpy.array([f['frame'] for f in track])
            bboxes = numpy.array([numpy.array(f['bbox']) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1);
            bboxesI = []
            for ij in range(0, 4): bboxesI.append(interp1d(frameNum, bboxes[:, ij])(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            tracks.append({'frame': frameI, 'bbox': bboxesI})
    return tracks


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


def get_smoothed_box_params(track):
    """This is the 'secret sauce' function we are investigating."""
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    # Apply the median filter
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    return dets


def main():
    parser = argparse.ArgumentParser(description="Investigate Bounding Box Smoothing")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file')
    args = parser.parse_args()

    # --- THIS IS THE FIX ---
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    # Get the root directory of the main project
    root_dir = script_dir.parent

    # Change the working directory to the script's location
    # This ensures all relative paths inside the LR-ASD code work correctly.
    os.chdir(script_dir)
    print(f"Changed working directory to: {os.getcwd()}")
    # --- END OF FIX ---

    video_path_abs = root_dir / args.video_path
    video_name = video_path_abs.stem
    temp_folder = root_dir / f"temp_investigation_{video_name}"

    if not temp_folder.exists():
        temp_folder.mkdir()

    print(f"Extracting frames to '{temp_folder}'...")
    command = (f'ffmpeg -y -i "{video_path_abs}" -t 4 -r 25 -f image2 "{temp_folder / "%06d.jpg"}" -loglevel quiet')
    subprocess.call(command, shell=True)

    raw_detections = inference_video(args, str(temp_folder))
    tracks = track_shot(raw_detections)

    if not tracks:
        print("\nCould not find a stable face track in the first few seconds. Investigation cannot proceed.")
        return

    print(f"\nFound {len(tracks)} stable face track(s). Analyzing the first one.")
    main_track = tracks[0]
    smoothed_params = get_smoothed_box_params(main_track)

    print("\n--- Bounding Box Comparison ---")
    print("Frame | Raw BBox (x1, y1, x2, y2)        | Smoothed BBox (x1, y1, x2, y2)")
    print("-------------------------------------------------------------------------")
    for i in range(min(10, len(main_track['frame']))):  # Show 10 frames for better comparison
        frame_index = main_track['frame'][i]
        raw_box = [int(p) for p in main_track['bbox'][i]]
        x, y, s = smoothed_params['x'][i], smoothed_params['y'][i], smoothed_params['s'][i]
        smoothed_box = [int(x - s), int(y - s), int(x + s), int(y + s)]
        print(f"{frame_index:^5} | {str(raw_box):<30} | {str(smoothed_box)}")

    print(f"\nInvestigation complete. The 'Smoothed BBox' is what creates the clean, centered box.")


if __name__ == "__main__":
    from tqdm import tqdm

    main()
