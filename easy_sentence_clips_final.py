#!/usr/bin/env python3
"""
easy_sentence_clips_final.py

One-shot pipeline:
 - Whisper transcription (sentence-level)
 - Run LR-ASD (Columbia_test.py)
 - Align Whisper sentences with LR-ASD tracks using global frame indices
 - Extract per-sentence face clips (from pycrop/*.avi or by cropping pyframes/)
 - Save clips and manifest.json

Usage:
    python easy_sentence_clips_final.py /path/to/video.mp4
"""
import sys, subprocess, json, math, re
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2
import pickle
import glob
import os

# whisper + torch
try:
    import whisper, torch
except Exception as e:
    print("ERROR: install whisper and torch. pip install -U openai-whisper torch")
    raise

# ------------- Utilities -------------

def run_whisper(video_path: Path, language="ja", model_size=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = model_size or ("medium" if torch.cuda.is_available() else "small")
    print(f"[whisper] loading '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)
    print("[whisper] transcribing...")
    result = model.transcribe(str(video_path),
                              language=language,
                              fp16=torch.cuda.is_available(),
                              word_timestamps=True,
                              verbose=False)
    segments = []
    for s in result.get("segments", []):
        segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text","").strip()})
    print(f"[whisper] {len(segments)} segments")
    return segments

def run_columbia(video_path: Path) -> Path:
    video_name = video_path.stem
    video_folder = str(video_path.parent)
    cmd = [sys.executable, "Columbia_test.py", "--videoName", video_name, "--videoFolder", video_folder]
    print("[LR-ASD] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return Path(video_folder) / video_name

def safe_load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

def detect_fps(base_dir: Path) -> float:
    cand = None
    for name in ("video.avi","video_only.avi","video_out.avi"):
        p = base_dir / "pyavi" / name
        if p.exists():
            cand = p; break
    if cand is None:
        raise FileNotFoundError("No pyavi video found to detect FPS")
    cap = cv2.VideoCapture(str(cand))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and fps>0:
        return float(fps)
    # fallback to ffprobe
    try:
        import shlex, subprocess
        out = subprocess.check_output(shlex.split(f"ffprobe -v 0 -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 \"{cand}\"")).decode().strip()
        if "/" in out:
            a,b = out.split("/")
            return float(a)/float(b)
        return float(out)
    except Exception:
        print("[warn] fps detection failed; default 25")
        return 25.0

# ------------- Track <-> score alignment helpers -------------

def build_tracks_and_scores(pywork_dir: Path):
    tracks_p = pywork_dir / "tracks.pckl"
    scores_p = pywork_dir / "scores.pckl"
    if not tracks_p.exists() or not scores_p.exists():
        raise FileNotFoundError("tracks.pckl or scores.pckl missing in pywork/")
    tracks = safe_load_pickle(tracks_p)
    scores = safe_load_pickle(scores_p)
    # Ensure both lists align by index (len same). If lengths mismatch, we'll handle defensively.
    if len(tracks) != len(scores):
        print(f"[warn] tracks ({len(tracks)}) vs scores ({len(scores)}) differ, proceeding defensively.")
    # convert scores -> numpy arrays
    scores_np = [np.asarray(s).flatten() if s is not None else np.array([]) for s in scores]
    return tracks, scores_np

def global_frames_overlap_range(track_frames: np.ndarray, start_f: int, end_f: int) -> np.ndarray:
    """Return local indices into track_frames that overlap start_f..end_f (inclusive)."""
    if track_frames.size == 0:
        return np.array([], dtype=int)
    idxs = np.where((track_frames >= start_f) & (track_frames <= end_f))[0]
    return idxs

# ------------- Speaker selection (robust) -------------

def choose_best_track_for_segment(tracks, scores, fps, seg_start, seg_end,
                                  min_frames=2, smoothing=3):
    start_f = int(math.floor(seg_start * fps))
    end_f   = int(math.ceil(seg_end * fps))
    best_idx = None
    best_val = -1e12

    for i, tr in enumerate(tracks):
        # track frames are global frame numbers
        try:
            track_frames = np.asarray(tr['track']['frame']).astype(int)
        except Exception:
            continue
        if track_frames.size == 0:
            continue
        score_arr = scores[i] if i < len(scores) else np.array([])
        if score_arr.size == 0:
            continue
        # compute local indices overlapping the segment
        local_idxs = global_frames_overlap_range(track_frames, start_f, end_f)
        if local_idxs.size < min_frames:
            continue
        # clip any local idxs that might be >= score_arr length (defensive)
        valid_local = local_idxs[local_idxs < score_arr.shape[0]]
        if valid_local.size == 0:
            continue
        sel = score_arr[valid_local]
        if smoothing > 1 and sel.size >= smoothing:
            k = np.ones(smoothing)/float(smoothing)
            sel = np.convolve(sel, k, mode='valid')
        val = float(np.mean(sel))
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx, best_val

# ------------- Extraction helpers -------------

def extract_from_pycrop_avi(pycrop_dir: Path, track_idx: int, seg_start: float, seg_end: float,
                            track_first_frame: int, fps: float, out_path: Path) -> bool:
    """
    Try to cut from a pycrop avi for this track.
    pycrop's avi often contains only local frames of that track; so we compute relative times.
    """
    # find avi file by numeric id in filename or by index order
    avi_candidates = sorted(list(pycrop_dir.glob("*.avi")) + list(pycrop_dir.glob("*/*.avi")))
    chosen = None
    # try filename numeric mapping
    for a in avi_candidates:
        m = re.search(r'(\d+)', a.stem)
        if m:
            if int(m.group(1)) == track_idx:
                chosen = a
                break
    if chosen is None and avi_candidates:
        # fallback to using track_idx order if within range
        if track_idx < len(avi_candidates):
            chosen = avi_candidates[track_idx]
        else:
            chosen = avi_candidates[0]
    if chosen is None:
        return False

    # compute relative time offset: track_first_frame / fps
    rel_start = seg_start - (track_first_frame / fps)
    rel_end   = seg_end   - (track_first_frame / fps)
    # clamp
    if rel_end <= 0:
        return False
    rel_start = max(0.0, rel_start)
    # check avi duration
    cap = cv2.VideoCapture(str(chosen))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    avifps = cap.get(cv2.CAP_PROP_FPS) or fps
    cap.release()
    avi_duration = total_frames / avifps if avifps>0 else 0.0
    if rel_start >= avi_duration:
        return False
    if rel_end > avi_duration:
        rel_end = avi_duration
    # run ffmpeg cut (try stream copy, fallback to re-encode)
    outp = str(out_path)
    cmd = ["ffmpeg", "-y", "-i", str(chosen), "-ss", f"{rel_start:.3f}", "-to", f"{rel_end:.3f}", "-c", "copy", outp]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        # fallback re-encode
        cmd2 = ["ffmpeg", "-y", "-i", str(chosen), "-ss", f"{rel_start:.3f}", "-to", f"{rel_end:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "128k", outp]
        try:
            subprocess.run(cmd2, check=True)
            return True
        except Exception:
            return False

def extract_from_pyframes_track(track_entry, pyframes_dir: Path, seg_start: float, seg_end: float,
                                fps: float, out_path: Path, pad_mode='center') -> bool:
    """
    Build mp4 by cropping pyframes using track bbox timeline for frames within seg_start..seg_end.
    """
    track_frames = np.asarray(track_entry['track']['frame']).astype(int)
    bboxes = np.asarray(track_entry['track']['bbox']).astype(int)
    if track_frames.size == 0 or bboxes.size == 0:
        return False
    start_f = int(math.floor(seg_start * fps))
    end_f   = int(math.ceil(seg_end * fps))
    idxs = global_frames_overlap_range(track_frames, start_f, end_f)
    if idxs.size == 0:
        return False
    # clip idxs to bbox array length (defensive)
    idxs = idxs[idxs < bboxes.shape[0]]
    if idxs.size == 0:
        return False
    widths = (bboxes[idxs,2] - bboxes[idxs,0]).clip(min=1)
    heights= (bboxes[idxs,3] - bboxes[idxs,1]).clip(min=1)
    target_w = int(min(1920, max(1, int(np.max(widths)))))
    target_h = int(min(1080, max(1, int(np.max(heights)))))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (target_w, target_h))
    if not writer.isOpened():
        print("[error] cv2 writer failed:", out_path)
        return False
    written = 0
    for idx in idxs:
        gf = int(track_frames[idx])
        frame_path = pyframes_dir / f"{gf:06d}.jpg"
        if not frame_path.exists():
            # fallback search brief
            cand = next(iter(pyframes_dir.glob(f"*{gf:03d}*.jpg")), None)
            if cand is None:
                continue
            frame_path = cand
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        x1,y1,x2,y2 = bboxes[idx]
        x1 = max(0, min(w_img-1, int(x1))); x2 = max(0, min(w_img, int(x2)))
        y1 = max(0, min(h_img-1, int(y1))); y2 = max(0, min(h_img, int(y2)))
        if x2<=x1 or y2<=y1: continue
        crop = img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        # place on canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if pad_mode == 'center':
            off_x = max(0, (target_w - cw)//2)
            off_y = max(0, (target_h - ch)//2)
        else:
            off_x = 0; off_y = 0
        ex = min(target_w, off_x + cw); ey = min(target_h, off_y + ch)
        sx = 0; sy = 0
        canvas[off_y:ey, off_x:ex] = crop[sy:sy+(ey-off_y), sx:sx+(ex-off_x)]
        writer.write(canvas)
        written += 1
    writer.release()
    return written > 0

# ------------- Main pipeline -------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python easy_sentence_clips_final.py /path/to/video.mp4")
        sys.exit(1)
    video_path = Path(sys.argv[1]).resolve()
    if not video_path.exists():
        print("Video not found:", video_path); sys.exit(1)

    # 1) Whisper
    segments = run_whisper(video_path, language="ja")

    # 2) Run LR-ASD
    base_out = run_columbia(video_path)
    if not base_out.exists():
        print("LR-ASD output folder missing:", base_out); sys.exit(1)
    pywork = base_out / "pywork"
    pyframes = base_out / "pyframes"
    pycrop = base_out / "pycrop"

    # 3) fps
    fps = detect_fps(base_out)
    print(f"[info] fps = {fps}")

    # 4) load tracks & scores
    tracks, scores = build_tracks_and_scores(pywork)
    print(f"[info] loaded {len(tracks)} tracks")

    # 5) optional pre-filter: drop tiny tracks
    min_track_len = 2
    keep_idx = [i for i,t in enumerate(tracks) if len(np.asarray(t['track']['frame']).flatten()) >= min_track_len]
    if len(keep_idx) < len(tracks):
        tracks = [tracks[i] for i in keep_idx]
        scores = [scores[i] for i in keep_idx]
        print(f"[info] filtered tiny tracks -> {len(tracks)} remain")

    # 6) map pycrop avi numeric indices (if present)
    avi_files = sorted(list(pycrop.glob("*.avi")) + list(pycrop.glob("*/*.avi")))
    avi_map = {}
    for a in avi_files:
        m = re.search(r'(\d+)', a.stem)
        if m:
            idx = int(m.group(1))
            avi_map[idx] = a

    # 7) process segments
    output_dir = base_out / "output"
    output_dir.mkdir(exist_ok=True)
    manifest = []

    for si, seg in enumerate(segments):
        sstart = float(seg["start"]); send = float(seg["end"]); text = seg["text"]
        print(f"[seg {si}] {sstart:.2f}-{send:.2f} : \"{text}\"")

        best_idx, best_val = choose_best_track_for_segment(tracks, scores, fps, sstart, send,
                                                           min_frames=2, smoothing=3)
        if best_idx is None:
            print("  -> no suitable track found; skipping")
            manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": None, "clip": None})
            continue

        tr = tracks[best_idx]
        # track first global frame
        track_frames = np.asarray(tr['track']['frame']).astype(int)
        if track_frames.size == 0:
            print("  -> chosen track has no frames; skipping")
            manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": int(best_idx), "clip": None})
            continue
        track_first = int(track_frames[0])

        out_clip = output_dir / f"seg_{si:03d}_trk{best_idx+1}.mp4"
        saved = False
        method = None

        # 7a) Try cutting from pycrop avi if mapping exists
        if best_idx in avi_map:
            avi_file = avi_map[best_idx]
            print(f"  trying pycrop avi {avi_file.name}")
            try:
                if extract_from_pycrop_avi(pycrop, best_idx, sstart, send, track_first, fps, out_clip):
                    saved = True; method = f"pycrop_avi:{avi_file.name}"
            except Exception as e:
                print("   pycrop avi cut failed:", e)

        # 7b) fallback: build from pyframes + track bbox
        if not saved:
            try:
                ok = extract_from_pyframes_track(tr, pyframes, sstart, send, fps, out_clip)
                if ok:
                    saved = True; method = "pyframes_crop"
            except Exception as e:
                print("   pyframes crop failed:", e)

        # 7c) fallback: if avi exists but not mapped by numeric id, try first avi file
        if not saved and avi_files:
            try:
                print("  trying first pycrop avi as fallback:", avi_files[0].name)
                if extract_from_pycrop_avi(pycrop, 0, sstart, send, track_first, fps, out_clip):
                    saved = True; method = f"pycrop_avi_fallback:{avi_files[0].name}"
            except Exception as e:
                pass

        if saved:
            print(f"  -> saved {out_clip} (method={method})")
            manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": int(best_idx), "score_val": float(best_val), "clip": str(out_clip), "method": method})
        else:
            print("  -> failed to create clip for this segment")
            manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": int(best_idx), "score_val": float(best_val), "clip": None, "method": None})

    # write manifest
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("[done] output dir:", output_dir)
    print("[done] manifest:", output_dir / "manifest.json")

if __name__ == "__main__":
    main()
