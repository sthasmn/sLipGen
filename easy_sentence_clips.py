#!/usr/bin/env python3
"""
easy_sentence_clips.py

Usage:
    python easy_sentence_clips.py <video_path>

What it does:
 - Transcribe the video with Whisper (sentence-level segments)
 - Run LR-ASD (Columbia_test.py) on the video to produce pywork/ (tracks.pckl, scores.pckl) and pyframes/
 - For each Whisper sentence, pick the active track (speaker) using per-track global-frame indices and scores
 - Build a cropped MP4 for that sentence from pyframes using the track's per-frame bbox
 - Save clips and manifest in: <video_folder>/<video_name>/output/

Notes:
 - This script assumes Columbia_test.py is in the same directory and can be run as `python Columbia_test.py --videoName <name> --videoFolder <folder>`
 - Works for "wild" videos because it uses global frame numbers from tracks.pckl to align tracks <-> timeline.
"""

import sys
import subprocess
import json
import pickle
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import glob

# whisper and torch
try:
    import whisper
    import torch
except Exception as e:
    print("ERROR: whisper or torch not installed. Install with: pip install -U openai-whisper torch")
    raise

# -------------------------
# Utilities
# -------------------------

def run_whisper_segments(video_path: Path, language="ja", model_size=None):
    """Return list of {start,end,text} segments from Whisper."""
    print("[whisper] loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = model_size or ("medium" if torch.cuda.is_available() else "small")
    print(f"[whisper] model='{model_size}' device={device}")
    model = whisper.load_model(model_size, device=device)
    print("[whisper] transcribing (this may take a while)...")
    result = model.transcribe(str(video_path),
                              language=language,
                              fp16=torch.cuda.is_available(),
                              word_timestamps=True,
                              verbose=False)
    segs = []
    for s in result.get("segments", []):
        segs.append({"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text","").strip()})
    print(f"[whisper] got {len(segs)} segments")
    return segs

def run_columbia(video_path: Path):
    """Run Columbia_test.py to produce LR-ASD outputs for the video."""
    video_name = video_path.stem
    video_folder = str(video_path.parent)
    cmd = [sys.executable, "Columbia_test.py", "--videoName", video_name, "--videoFolder", video_folder]
    print("[LR-ASD] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return Path(video_folder) / video_name

def safe_load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

def get_fps_from_pyavi(base_dir: Path):
    """Detect FPS from pyavi video files using OpenCV, fallback to ffprobe if needed."""
    candidates = [
        base_dir / "pyavi" / "video.avi",
        base_dir / "pyavi" / "video_only.avi",
        base_dir / "pyavi" / "video_out.avi"
    ]
    for c in candidates:
        if c.exists():
            cap = cv2.VideoCapture(str(c))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0:
                return float(fps)
    # fallback to ffprobe if available
    try:
        import shlex, subprocess
        c = candidates[0]
        cmd = f"ffprobe -v 0 -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 \"{c}\""
        out = subprocess.check_output(shlex.split(cmd)).decode().strip()
        if "/" in out:
            a,b = out.split("/")
            return float(a)/float(b)
        else:
            return float(out)
    except Exception:
        print("[warn] could not detect fps; defaulting to 25.0")
        return 25.0

# -------------------------
# Core: choose speaker using per-track frames & scores
# -------------------------

def choose_speaker_for_segment_using_tracks(tracks_list, scores_list, fps, seg_start, seg_end,
                                           min_frames_for_consider=1, smooth_win=3):
    """
    tracks_list: list from tracks.pckl (each item contains track['track']['frame'] and track['track']['bbox'])
    scores_list: list parallel to tracks_list with ASD scores per-track (one value per frame of that track)
    fps: frames per second
    seg_start, seg_end: seconds
    Returns: track_index (int) or None
    """
    start_f = int(math.floor(seg_start * fps))
    end_f   = int(math.ceil(seg_end * fps))
    best_track = None
    best_score = -1e12

    for trk_idx, tr in enumerate(tracks_list):
        try:
            frame_arr = np.asarray(tr['track']['frame']).astype(int)
        except Exception:
            continue
        if frame_arr.size == 0:
            continue
        score_arr = np.asarray(scores_list[trk_idx]).flatten()
        if score_arr.size == 0:
            continue
        # find indices within this track whose global frames fall into segment range
        idxs = np.where((frame_arr >= start_f) & (frame_arr <= end_f))[0]
        if idxs.size < min_frames_for_consider:
            continue
        sel = score_arr[idxs]
        if smooth_win > 1 and sel.size >= smooth_win:
            # moving average smoothing
            kernel = np.ones(smooth_win)/float(smooth_win)
            sel = np.convolve(sel, kernel, mode='valid')
        mean_score = float(np.mean(sel))
        if mean_score > best_score:
            best_score = mean_score
            best_track = trk_idx
    return best_track

# -------------------------
# Core: build cropped clip from pyframes using track bbox timeline
# -------------------------

def extract_cropped_clip_from_track(track_entry, pyframes_dir: Path, seg_start, seg_end, out_path: Path, fps=25,
                                    pad_mode='center', bgcolor=(0,0,0)):
    """
    track_entry: one element from tracks.pckl
    pyframes_dir: folder with frame images like 000001.jpg
    seg_start/seg_end: seconds
    out_path: Path to output mp4
    fps: fps to map seconds -> frame index
    """
    # extract arrays
    track_frames = np.asarray(track_entry['track']['frame']).astype(int)
    bboxes = np.asarray(track_entry['track']['bbox']).astype(int)
    if track_frames.size == 0 or bboxes.size == 0:
        return False

    start_f = int(math.floor(seg_start * fps))
    end_f = int(math.ceil(seg_end * fps))
    # find frame indices within the track that overlap the segment
    idxs = np.where((track_frames >= start_f) & (track_frames <= end_f))[0]
    if idxs.size == 0:
        return False

    # compute canvas size (max bbox size across selected frames)
    widths = (bboxes[idxs, 2] - bboxes[idxs, 0]).clip(min=1)
    heights = (bboxes[idxs, 3] - bboxes[idxs, 1]).clip(min=1)
    target_w = int(min(1920, int(np.max(widths))))
    target_h = int(min(1080, int(np.max(heights))))
    if target_w <=0 or target_h <=0:
        return False

    # setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (target_w, target_h))
    if not writer.isOpened():
        print("[error] cannot open VideoWriter for", out_path)
        return False

    # iterate through selected track frames in chronological order
    for idx in idxs:
        frame_num = int(track_frames[idx])
        fname = pyframes_dir / f"{frame_num:06d}.jpg"
        if not fname.exists():
            # fallback: try pattern search (slower)
            candidates = sorted(pyframes_dir.glob(f"*{frame_num:03d}*.jpg"))
            if candidates:
                fname = candidates[0]
            else:
                # skip missing frames silently
                continue
        img = cv2.imread(str(fname))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        x1, y1, x2, y2 = bboxes[idx]
        x1 = max(0, min(w_img-1, int(x1)))
        x2 = max(0, min(w_img, int(x2)))
        y1 = max(0, min(h_img-1, int(y1)))
        y2 = max(0, min(h_img, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]

        # place onto canvas
        canvas = np.full((target_h, target_w, 3), bgcolor, dtype=np.uint8)
        if pad_mode == 'center':
            off_x = max(0, (target_w - cw)//2)
            off_y = max(0, (target_h - ch)//2)
        else:
            off_x = 0; off_y = 0
        ex = min(target_w, off_x + cw)
        ey = min(target_h, off_y + ch)
        sx = 0; sy = 0
        canvas[off_y:ey, off_x:ex] = crop[sy:sy+(ey-off_y), sx:sx+(ex-off_x)]
        writer.write(canvas)

    writer.release()
    return True

# -------------------------
# Small helpers for pycrop fallback detection
# -------------------------

def find_pycrop_avi_files(pycrop_dir: Path):
    """Return sorted avi files directly under pycrop or within its subfolders."""
    if not pycrop_dir.exists():
        return []
    files = sorted(pycrop_dir.glob("*.avi"))
    if files:
        return files
    nested = sorted(pycrop_dir.glob("*/*.avi"))
    return nested

def frames_folder_for_track(pycrop_dir: Path, track_idx_onebased: int):
    # try 6-digit zero padded folder
    cand = pycrop_dir / f"{track_idx_onebased:06d}"
    if cand.exists() and cand.is_dir():
        return cand
    cand2 = pycrop_dir / f"{track_idx_onebased}"
    if cand2.exists() and cand2.is_dir():
        return cand2
    # fallback: pick nth folder
    folders = sorted([p for p in pycrop_dir.iterdir() if p.is_dir()])
    if folders and (track_idx_onebased-1) < len(folders):
        return folders[track_idx_onebased-1]
    return None

# -------------------------
# Main pipeline
# -------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python easy_sentence_clips.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1]).resolve()
    if not video_path.exists():
        print("ERROR: video not found:", video_path)
        sys.exit(1)

    # 1) transcribe
    segments = run_whisper_segments(video_path, language="ja")

    # 2) run LR-ASD to produce pywork/ pyframes/ pycrop/
    base_out = run_columbia(video_path)  # returns folder path like <parent>/<video_name>
    if not base_out.exists():
        print("ERROR: expected LR-ASD output folder not found:", base_out)
        sys.exit(1)

    pywork = base_out / "pywork"
    pyframes = base_out / "pyframes"
    pycrop = base_out / "pycrop"

    # 3) fps detection
    fps = get_fps_from_pyavi(base_out)
    print(f"[info] using fps = {fps}")

    # 4) load tracks & scores
    tracks_pkl = pywork / "tracks.pckl"
    scores_pkl = pywork / "scores.pckl"
    if not tracks_pkl.exists() or not scores_pkl.exists():
        print("ERROR: tracks.pckl or scores.pckl missing in", pywork)
        sys.exit(1)

    print("[info] loading tracks.pckl and scores.pckl ...")
    tracks_list = safe_load_pickle(tracks_pkl)
    scores_list = safe_load_pickle(scores_pkl)
    # ensure arrays
    scores_list = [np.asarray(s).flatten() for s in scores_list]
    print(f"[info] loaded {len(tracks_list)} tracks and {len(scores_list)} score arrays")

    # optionally filter out super-short tracks (noise)
    min_track_frames = 3  # configurable
    filtered_indices = [i for i,tr in enumerate(tracks_list) if len(np.asarray(tr['track']['frame']).flatten()) >= min_track_frames]
    if len(filtered_indices) < len(tracks_list):
        print(f"[info] filtering out {len(tracks_list)-len(filtered_indices)} very short tracks (<{min_track_frames} frames)")
        # create filtered lists
        tracks_list = [tracks_list[i] for i in filtered_indices]
        scores_list = [scores_list[i] for i in filtered_indices]

    # 5) prepare output dir
    output_dir = base_out / "output"
    output_dir.mkdir(exist_ok=True)

    manifest = []

    # 6) for each whisper segment, choose speaker and extract clip
    for idx, seg in enumerate(segments):
        sstart = float(seg["start"]); send = float(seg["end"]); text = seg["text"]
        print(f"[segment {idx}] {sstart:.2f}-{send:.2f} : \"{text}\"")

        best_track = choose_speaker_for_segment_using_tracks(tracks_list, scores_list, fps, sstart, send,
                                                            min_frames_for_consider=2, smooth_win=3)
        if best_track is None:
            print(f"  -> No candidate track found. skipping.")
            manifest.append({"index": idx, "start": sstart, "end": send, "text": text, "speaker": None, "clip": None})
            continue

        # Try to build clip from pyframes using that track's bbox timeline
        out_file = output_dir / f"seg_{idx:03d}_trk{best_track+1}.mp4"
        ok = extract_cropped_clip_from_track(tracks_list[best_track], pyframes, sstart, send, out_file, fps=fps)
        method = "pyframes_crop"

        # If that failed, try pycrop avi (if present)
        if not ok:
            avi_files = find_pycrop_avi_files(pycrop)
            if avi_files:
                # map by order: if number of avi >= tracks, try index
                if len(avi_files) >= (best_track+1):
                    try_avi = avi_files[best_track]
                else:
                    try_avi = avi_files[0]
                print(f"  -> trying cut from pycrop avi {try_avi.name}")
                # cut by segment: we must map seconds relative to original video (pycrop avi is usually aligned)
                try:
                    # use ffmpeg to cut
                    cmd = [
                        "ffmpeg","-y","-i", str(try_avi),
                        "-ss", f"{sstart:.3f}", "-to", f"{send:.3f}",
                        "-c","copy", str(out_file)
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    ok = True
                    method = f"pycrop_avi:{try_avi.name}"
                except Exception:
                    ok = False

        if not ok:
            # As final fallback, try frames_folder_for_track
            frames_folder = frames_folder_for_track(pycrop, best_track+1)
            if frames_folder:
                ok = extract_cropped_clip_from_track(tracks_list[best_track], frames_folder, sstart, send, out_file, fps=fps)
                method = f"pycrop_frames:{frames_folder.name}" if ok else method

        if ok:
            print(f"  -> saved clip {out_file} (method={method})")
            manifest.append({"index": idx, "start": sstart, "end": send, "text": text, "speaker_track": int(best_track), "clip": str(out_file), "method": method})
        else:
            print(f"  -> failed to create clip for track {best_track}")
            manifest.append({"index": idx, "start": sstart, "end": send, "text": text, "speaker_track": int(best_track), "clip": None})

    # 7) write manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("[done] output saved to:", output_dir)
    print("[done] manifest:", manifest_path)

if __name__ == "__main__":
    main()
