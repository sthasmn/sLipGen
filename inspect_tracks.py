#!/usr/bin/env python3
# inspect_tracks.py
import sys, argparse, pickle
from pathlib import Path
import numpy as np
import cv2
import json, subprocess

def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def get_fps(base_dir):
    cands = [
        base_dir/"pyavi"/"video.avi",
        base_dir/"pyavi"/"video_only.avi",
        base_dir/"pyavi"/"video_out.avi"
    ]
    for c in cands:
        if c.exists():
            cap = cv2.VideoCapture(str(c))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0:
                return float(fps)
    # fallback try ffprobe
    for c in cands:
        if c.exists():
            try:
                import shlex, subprocess
                cmd = f'ffprobe -v 0 -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{c}"'
                out = subprocess.check_output(shlex.split(cmd)).decode().strip()
                if "/" in out:
                    a,b = out.split("/")
                    return float(a)/float(b)
                else:
                    return float(out)
            except Exception as e:
                break
    return 25.0

def inspect(base_dir, seg_index=None, seg_start=None, seg_end=None, verbose=True):
    base_dir = Path(base_dir)
    pywork = base_dir/"pywork"
    pyframes = base_dir/"pyframes"
    pycrop = base_dir/"pycrop"
    print("Base dir:", base_dir)
    print("pywork:", pywork.exists(), "pyframes:", pyframes.exists(), "pycrop:", pycrop.exists())

    tracks_p = pywork/"tracks.pckl"
    scores_p = pywork/"scores.pckl"
    if not tracks_p.exists() or not scores_p.exists():
        print("Missing tracks.pckl or scores.pckl in", pywork)
        return

    tracks = load_pickle(tracks_p)
    scores = load_pickle(scores_p)

    n_tracks = len(tracks)
    n_scores = len(scores)
    print(f"Loaded tracks: {n_tracks} , scores arrays: {n_scores}")

    fps = get_fps(base_dir)
    print("Detected fps:", fps)

    # Check per-track lengths & frame ranges
    mismatches = []
    lengths = []
    ranges = []
    for i, tr in enumerate(tracks):
        frame_arr = np.asarray(tr['track']['frame']).astype(int)
        bbox_arr = np.asarray(tr['track']['bbox'])
        score_arr = np.asarray(scores[i]).flatten() if i < len(scores) else None
        lengths.append(len(frame_arr))
        if score_arr is not None:
            if len(score_arr) != len(frame_arr):
                mismatches.append((i, len(frame_arr), len(score_arr)))
        if frame_arr.size:
            ranges.append((i, int(frame_arr[0]), int(frame_arr[-1])))
    print("Tracks: min/median/max lengths:", np.min(lengths), int(np.median(lengths)), np.max(lengths))
    print("Number of tracks with len(frame) != len(score):", len(mismatches))
    if mismatches and verbose:
        print("First 10 mismatches (track_idx, frames_len, score_len):")
        for m in mismatches[:10]:
            print(" ", m)

    # histogram of track start frames
    starts = [r[1] for r in ranges] if ranges else []
    if starts:
        print("Global frame starts: min/median/max:", min(starts), int(np.median(starts)), max(starts))

    # if a segment provided by index, load whisper segments manifest if exists
    if seg_index is not None:
        # try to load existing manifest or produce approximate time by user input
        # The script does not run whisper; read user-provided --start/--end if given
        pass

    # If segment start/end specified, inspect candidate tracks overlapping that interval
    if seg_start is not None and seg_end is not None:
        start_f = int(np.floor(seg_start * fps))
        end_f = int(np.ceil(seg_end * fps))
        print(f"\nInspecting interval {seg_start:.3f}s - {seg_end:.3f}s => frames {start_f} .. {end_f}")
        print("Tracks overlapping that interval (showing up to 50):")
        count = 0
        for i, tr in enumerate(tracks):
            frame_arr = np.asarray(tr['track']['frame']).astype(int)
            if frame_arr.size == 0:
                continue
            # overlap test
            if frame_arr[-1] < start_f or frame_arr[0] > end_f:
                continue
            # compute local index range
            idxs = np.where((frame_arr >= start_f) & (frame_arr <= end_f))[0]
            score_len = len(scores[i]) if i < len(scores) else None
            print(f" Track {i}: global_frames {frame_arr[0]}..{frame_arr[-1]}  total={len(frame_arr)}  overlap_frames={len(idxs)}  score_len={score_len}")
            # show a sample of indices
            if len(idxs)>0:
                print("   sample local idxs:", idxs[:5], "...", idxs[-3:])
                # check if these local idxs are within score range
                if score_len is not None:
                    if idxs.size>0 and idxs.max() >= score_len:
                        print("   !!! WARNING: local idx max", idxs.max(), ">= score_len", score_len)
            # check frame files exist
            if len(idxs) > 0:
                missing = 0
                for idx in idxs[:20]:
                    gf = frame_arr[idx]
                    fpath = pyframes / f"{gf:06d}.jpg"
                    if not fpath.exists():
                        missing += 1
                if missing:
                    print("   Note: some pyframes are missing (checked first 20 overlaps):", missing)
            count += 1
            if count >= 50:
                break

    # Check pycrop avi files (durations)
    avi_files = sorted(list(pycrop.glob("*.avi")) + list(pycrop.glob("*/*.avi")))
    if avi_files:
        print("\npycrop avi files found (name, frames, fps, duration_sec):")
        for a in avi_files[:50]:
            cap = cv2.VideoCapture(str(a))
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            afps = cap.get(cv2.CAP_PROP_FPS) or fps
            cap.release()
            dur = nframes / afps if afps>0 else 0.0
            print(" ", a.name, nframes, afps, f"{dur:.2f}s")
    else:
        print("\nNo pycrop .avi found under", pycrop)

    print("\nDone inspection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="path to video file (used to find base output folder <parent>/<stem>)")
    parser.add_argument("--segment", type=int, help="whisper segment index (optional)")
    parser.add_argument("--start", type=float, help="start time in seconds to inspect (optional)")
    parser.add_argument("--end", type=float, help="end time in seconds to inspect (optional)")
    args = parser.parse_args()

    base = Path(args.video).resolve().parent / Path(args.video).resolve().stem
    if not base.exists():
        print("Base output folder does not exist:", base)
        sys.exit(1)
    seg_idx = args.segment
    seg_start = args.start
    seg_end = args.end
    # if only segment given, try to read manifest or whisper output (none available), so user should pass start/end
    inspect(base, seg_index=seg_idx, seg_start=seg_start, seg_end=seg_end)

