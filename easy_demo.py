import os
import sys
import pickle
import cv2
import numpy as np

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def annotate_video(video_dir):
    pyavi = os.path.join(video_dir, "pyavi", "video_only.avi")
    pywork = os.path.join(video_dir, "pywork")

    # Load LR-ASD results
    faces = load_pickle(os.path.join(pywork, "faces.pckl"))
    tracks = load_pickle(os.path.join(pywork, "tracks.pckl"))
    scores = load_pickle(os.path.join(pywork, "scores.pckl"))

    # Open video
    cap = cv2.VideoCapture(pyavi)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video
    out_path = os.path.join(video_dir, "final_output.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw tracks
        if frame_idx in tracks:
            for tid, track in tracks[frame_idx].items():
                x1, y1, x2, y2 = track["bbox"]
                speaker_prob = scores.get((frame_idx, tid), 0)

                # Choose color based on speaking status
                if speaker_prob > 0.5:
                    color = (0, 255, 0)  # green = speaking
                    label = f"Speaker {tid} (ON)"
                else:
                    color = (0, 0, 255)  # red = silent
                    label = f"Speaker {tid}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] Final annotated video saved at: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python easy_demo.py <video_dir>")
        sys.exit(1)

    video_dir = sys.argv[1]
    annotate_video(video_dir)
