#!/usr/bin/env python3
"""
easy_sentence_clips_skipasd_audio_sync.py

Skip LR-ASD run (assumes LR-ASD outputs already exist).
Transcribe pyavi/audio.wav (preferred) so timestamps line up with LR-ASD outputs.
Crop pyframes per chosen track for each sentence and mux the corresponding pyavi audio.

Usage:
    python easy_sentence_clips_skipasd_audio_sync.py /path/to/original_video.mp4

Notes:
 - Requires: whisper, torch, numpy, opencv-python, ffmpeg available in PATH.
 - If pyavi/audio.wav exists it will be used for transcription and audio extraction.
 - If not found, the script will fall back to transcribing the provided input video file (risk of small mismatch).
"""
import sys, subprocess, json, math, tempfile, os, shutil, re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import pickle

# whisper + torch
try:
    import whisper, torch
except Exception as e:
    print("ERROR: install whisper and torch: pip install -U openai-whisper torch")
    raise

# ----------------- Utilities -----------------

def safe_load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

def detect_fps(base_dir: Path) -> float:
    """Detect fps from pyavi video files; fallback 25."""
    for name in ("video.avi","video_only.avi","video_out.avi"):
        p = base_dir / "pyavi" / name
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0:
                return float(fps)
    return 25.0

def ffprobe_duration(path: Path) -> float:
    """Return media duration in seconds using ffprobe; fallback 0."""
    try:
        import shlex, subprocess
        out = subprocess.check_output(shlex.split(f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{path}"'))
        return float(out.decode().strip())
    except Exception:
        return 0.0

def run_whisper_on_audio(audio_path: Path, model_size: Optional[str]=None, language: str="ja"):
    """Run whisper on an audio (or video) file; returns list of segments {start,end,text}."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = model_size or ("large-v2" if torch.cuda.is_available() else "small")
    print(f"[Whisper] loading {model_size} on {device} ...")
    model = whisper.load_model(model_size, device=device)
    print(f"[Whisper] transcribing {audio_path} ...")
    res = model.transcribe(str(audio_path), language=language, fp16=torch.cuda.is_available(), word_timestamps=True, verbose=False)
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text","").strip()} for s in res.get("segments",[])]
    print(f"[Whisper] {len(segs)} segments")
    return segs

# ----------------- Track / score helpers -----------------

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
    """Return local indices into the track arrays overlapping global frames start_f..end_f (inclusive)."""
    if track_frames.size == 0:
        return np.array([], dtype=int)
    return np.where((track_frames >= start_f) & (track_frames <= end_f))[0]

def choose_best_track(tracks, scores, fps: float, seg_start: float, seg_end: float, min_frames=2, smoothing=3) -> Tuple[Optional[int], Optional[float]]:
    """Return the best track index and its mean score value for the segment, or (None,None)."""
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
        local_idxs = local_idxs[local_idxs < score_arr.shape[0]]  # defensive
        if local_idxs.size == 0: continue
        sel = score_arr[local_idxs]
        if smoothing > 1 and sel.size >= smoothing:
            kernel = np.ones(smoothing)/float(smoothing)
            sel = np.convolve(sel, kernel, mode='valid')
        val = float(np.mean(sel))
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx, (best_val if best_idx is not None else None)

# ----------------- Extraction + audio muxing -----------------

def extract_cropped_video_from_pyframes(track_entry, pyframes_dir: Path, seg_start: float, seg_end: float,
                                        fps: float, out_video_path: Path, pad_mode='center') -> bool:
    """
    Crop frames from pyframes using track bboxes and write an MP4 (no audio).
    Returns True if at least one frame was written.
    """
    track_frames = np.asarray(track_entry['track']['frame']).astype(int)
    bboxes = np.asarray(track_entry['track']['bbox']).astype(int)
    if track_frames.size == 0 or bboxes.size == 0: return False
    start_f = int(math.floor(seg_start * fps)); end_f = int(math.ceil(seg_end * fps))
    idxs = overlap_local_indices(track_frames, start_f, end_f)
    if idxs.size == 0: return False
    # clamp idxs to bbox length
    idxs = idxs[idxs < bboxes.shape[0]]
    if idxs.size == 0: return False
    widths = (bboxes[idxs,2] - bboxes[idxs,0]).clip(min=1)
    heights= (bboxes[idxs,3] - bboxes[idxs,1]).clip(min=1)
    target_w = int(min(1920, int(np.max(widths))))
    target_h = int(min(1080, int(np.max(heights))))
    if target_w <= 0 or target_h <= 0: return False
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (target_w, target_h))
    if not writer.isOpened(): return False
    written = 0
    for idx in idxs:
        gf = int(track_frames[idx])
        frame_file = pyframes_dir / f"{gf:06d}.jpg"
        if not frame_file.exists():
            # fallback quick glob
            matches = list(pyframes_dir.glob(f"*{gf:03d}*.jpg"))
            frame_file = matches[0] if matches else None
            if frame_file is None: continue
        img = cv2.imread(str(frame_file))
        if img is None: continue
        h_img, w_img = img.shape[:2]
        x1,y1,x2,y2 = bboxes[idx]
        x1 = max(0, min(w_img-1, int(x1))); x2 = max(0, min(w_img, int(x2)))
        y1 = max(0, min(h_img-1, int(y1))); y2 = max(0, min(h_img, int(y2)))
        if x2<=x1 or y2<=y1: continue
        crop = img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if pad_mode == 'center':
            off_x = max(0, (target_w - cw)//2); off_y = max(0, (target_h - ch)//2)
        else:
            off_x = 0; off_y = 0
        ex = min(target_w, off_x + cw); ey = min(target_h, off_y + ch)
        canvas[off_y:ey, off_x:ex] = crop[0:(ey-off_y), 0:(ex-off_x)]
        writer.write(canvas)
        written += 1
    writer.release()
    return written > 0

def extract_audio_segment_from_pyavi(pyavi_audio: Path, seg_start: float, seg_end: float, out_audio: Path) -> bool:
    """
    Extract audio segment from pyavi/audio.wav (or pyavi/video.avi) and encode as AAC for muxing.
    """
    if not pyavi_audio.exists():
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", str(pyavi_audio),
        "-ss", f"{seg_start:.3f}", "-to", f"{seg_end:.3f}",
        "-vn", "-acodec", "aac", "-b:a", "128k", str(out_audio)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def mux_audio_into_video(video_path: Path, audio_path: Path, out_path: Path) -> bool:
    """
    Mux the given audio (AAC) into the video (copy video stream).
    """
    if not video_path.exists() or not audio_path.exists():
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

# ----------------- Main workflow -----------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python easy_sentence_clips_skipasd_audio_sync.py /path/to/original_video.mp4")
        sys.exit(1)
    orig_video = Path(sys.argv[1]).resolve()
    if not orig_video.exists():
        print("Original video not found:", orig_video); sys.exit(1)
    base = orig_video.parent / orig_video.stem
    if not base.exists():
        print("Expected LR-ASD output folder not found:", base); sys.exit(1)

    pywork = base / "pywork"
    pyframes = base / "pyframes"
    pyavi_dir = base / "pyavi"
    output_dir = base / "output"
    output_dir.mkdir(exist_ok=True)

    # prefer pyavi audio
    pyavi_audio = pyavi_dir / "audio.wav"
    pyavi_video = pyavi_dir / "video.avi"
    use_pyavi_audio = False
    audio_for_transcription = None
    if pyavi_audio.exists():
        audio_for_transcription = pyavi_audio
        use_pyavi_audio = True
        print("[info] Found pyavi/audio.wav - using it for transcription and audio extraction.")
    elif pyavi_video.exists():
        audio_for_transcription = pyavi_video
        use_pyavi_audio = True
        print("[info] pyavi/video.avi present. Will transcribe its audio.")
    else:
        audio_for_transcription = orig_video
        print("[warn] pyavi audio not found - will transcribe original video (may cause timestamp mismatch).")

    # Transcribe — prefer pyavi audio
    segments = run_whisper_on_audio(audio_for_transcription, model_size=None, language="ja")

    # load tracks + scores
    tracks, scores = load_tracks_scores(pywork)
    fps = detect_fps(base)
    print(f"[info] fps (pyavi) = {fps}")

    # Optional: Print durations to detect mismatch
    dur_orig = ffprobe_duration(orig_video)
    dur_pyavi = ffprobe_duration(pyavi_video) if pyavi_video.exists() else 0.0
    if dur_pyavi and abs(dur_orig - dur_pyavi) > 0.05:
        print(f"[warn] duration mismatch: original={dur_orig:.2f}s pyavi={dur_pyavi:.2f}s")

    manifest = []

    for si, seg in enumerate(segments):
        sstart = float(seg["start"]); send = float(seg["end"]); text = seg["text"]
        print(f"[seg {si:03d}] {sstart:.3f}-{send:.3f} : {text[:60]!s}")

        best_idx, best_val = choose_best_track(tracks, scores, fps, sstart, send, min_frames=2, smoothing=3)
        if best_idx is None:
            print("  -> no suitable track found")
            manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": None, "clip": None})
            continue

        track_entry = tracks[best_idx]
        # 1) produce cropped video from pyframes (no audio) into tmp
        tmp_vid = output_dir / f"tmp_seg_{si:03d}_trk{best_idx+1}.mp4"
        final_vid = output_dir / f"seg_{si:03d}_trk{best_idx+1}.mp4"
        final_vid_audio = output_dir / f"seg_{si:03d}_trk{best_idx+1}_audio.mp4"

        ok_vid = extract_cropped_video_from_pyframes(track_entry, pyframes, sstart, send, fps, tmp_vid)
        if not ok_vid:
            print("  -> failed to build cropped video from pyframes")
            manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": int(best_idx), "clip": None, "score": best_val})
            continue

        # 2) extract audio from pyavi (preferred) into tmp_aac
        tmp_aac = output_dir / f"tmp_seg_{si:03d}.aac"
        audio_source = pyavi_audio if pyavi_audio.exists() else (pyavi_video if pyavi_video.exists() else orig_video)
        ok_audio = extract_audio_segment_from_pyavi(audio_source, sstart, send, tmp_aac)
        if not ok_audio:
            # try extracting from original video (final fallback)
            ok_audio = extract_audio_segment_from_pyavi(orig_video, sstart, send, tmp_aac)
        if not ok_audio:
            print("  -> failed to extract audio for segment; saving video-only")
            shutil.move(str(tmp_vid), str(final_vid))
            manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": int(best_idx), "clip": str(final_vid), "clip_audio": None, "score": best_val})
            continue

        # 3) mux audio into tmp_vid => final_vid_audio
        ok_mux = mux_audio_into_video(tmp_vid, tmp_aac, final_vid_audio)
        if ok_mux:
            # delete tmp files
            tmp_vid.unlink(missing_ok=True)
            tmp_aac.unlink(missing_ok=True)
            print(f"  -> saved {final_vid_audio}")
            manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": int(best_idx), "clip": str(final_vid_audio), "score": float(best_val)})
        else:
            # fallback: keep video-only
            tmp_aac.unlink(missing_ok=True)
            shutil.move(str(tmp_vid), str(final_vid))
            print(f"  -> mux failed, saved video-only {final_vid}")
            manifest.append({"index": si, "start": sstart, "end": send, "text": text, "track": int(best_idx), "clip": str(final_vid), "score": float(best_val)})

    # write manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("[done] outputs:", output_dir)
    print("[done] manifest:", manifest_path)

if __name__ == "__main__":
    main()
