#!/usr/bin/env python3
"""
easy_sentence_clips_skipasd.py

Pipeline without running LR-ASD again:
 - Whisper transcription (sentence-level)
 - Load existing pywork/, pyframes/, pycrop/
 - Align Whisper sentences with LR-ASD tracks
 - Extract per-sentence face clips
 - Save clips + manifest.json

Usage:
    python easy_sentence_clips_skipasd.py /path/to/video.mp4
"""
import sys, json, math, re, subprocess
from pathlib import Path
import numpy as np
import cv2, pickle
import whisper, torch

# ------------ utils ------------
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
    segments = [{"start": float(s["start"]),
                 "end": float(s["end"]),
                 "text": s.get("text","").strip()}
                for s in result.get("segments", [])]
    print(f"[whisper] {len(segments)} segments")
    return segments

def safe_load_pickle(p: Path):
    with open(p, "rb") as f: return pickle.load(f)

def detect_fps(base_dir: Path) -> float:
    cand = None
    for name in ("video.avi","video_only.avi","video_out.avi"):
        p = base_dir / "pyavi" / name
        if p.exists(): cand = p; break
    if cand is None: return 25.0
    cap = cv2.VideoCapture(str(cand))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps>0 else 25.0

# ------------ track alignment ------------
def build_tracks_and_scores(pywork_dir: Path):
    tracks = safe_load_pickle(pywork_dir/"tracks.pckl")
    scores = safe_load_pickle(pywork_dir/"scores.pckl")
    scores = [np.asarray(s).flatten() if s is not None else np.array([]) for s in scores]
    return tracks, scores

def overlap_idxs(track_frames: np.ndarray, start_f: int, end_f: int):
    return np.where((track_frames >= start_f) & (track_frames <= end_f))[0]

def choose_best_track(tracks, scores, fps, seg_start, seg_end, min_frames=2):
    start_f = int(math.floor(seg_start*fps)); end_f = int(math.ceil(seg_end*fps))
    best_idx, best_val = None, -1e9
    for i,tr in enumerate(tracks):
        tf = np.asarray(tr['track']['frame']).astype(int)
        if tf.size==0: continue
        sc = scores[i] if i<len(scores) else np.array([])
        if sc.size==0: continue
        idxs = overlap_idxs(tf, start_f, end_f)
        idxs = idxs[idxs<sc.shape[0]]
        if idxs.size<min_frames: continue
        val = float(np.mean(sc[idxs]))
        if val>best_val: best_val, best_idx = val, i
    return best_idx, best_val

# ------------ extraction ------------
def extract_from_pyframes(tr, pyframes_dir, seg_start, seg_end, fps, out_path):
    tf = np.asarray(tr['track']['frame']).astype(int)
    bbs= np.asarray(tr['track']['bbox']).astype(int)
    if tf.size==0: return False
    start_f = int(seg_start*fps); end_f = int(seg_end*fps)
    idxs = overlap_idxs(tf, start_f, end_f)
    if idxs.size==0: return False
    idxs = idxs[idxs< bbs.shape[0]]
    if idxs.size==0: return False
    w = int(np.max(bbs[idxs,2]-bbs[idxs,0])); h = int(np.max(bbs[idxs,3]-bbs[idxs,1]))
    w=max(1,min(1920,w)); h=max(1,min(1080,h))
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(str(out_path),fourcc,fps,(w,h))
    for idx in idxs:
        gf=tf[idx]; fpath=pyframes_dir/f"{gf:06d}.jpg"
        if not fpath.exists(): continue
        img=cv2.imread(str(fpath));
        if img is None: continue
        x1,y1,x2,y2= bbs[idx]
        crop=img[y1:y2,x1:x2];
        if crop.size==0: continue
        crop=cv2.resize(crop,(w,h))
        out.write(crop)
    out.release()
    return True

# ------------ main ------------
def main():
    if len(sys.argv)<2:
        print("Usage: python easy_sentence_clips_skipasd.py video.mp4"); sys.exit(1)
    video_path=Path(sys.argv[1]).resolve()
    base_out=video_path.parent/video_path.stem
    pywork=base_out/"pywork"; pyframes=base_out/"pyframes"; pycrop=base_out/"pycrop"
    if not pywork.exists():
        print("Missing pywork/. Run Columbia_test.py once first."); sys.exit(1)

    segs=run_whisper(video_path)
    fps=detect_fps(base_out)
    tracks,scores=build_tracks_and_scores(pywork)
    output_dir=base_out/"output"; output_dir.mkdir(exist_ok=True)
    manifest=[]
    for i,s in enumerate(segs):
        best,score=choose_best_track(tracks,scores,fps,s["start"],s["end"])
        if best is None:
            manifest.append({"index":i,"text":s["text"],"clip":None}); continue
        out_clip=output_dir/f"seg_{i:03d}_trk{best+1}.mp4"
        ok=extract_from_pyframes(tracks[best],pyframes,s["start"],s["end"],fps,out_clip)
        manifest.append({"index":i,"text":s["text"],"clip":str(out_clip) if ok else None,"track":int(best),"score":score})
        print(f"[seg {i}] -> {'saved' if ok else 'failed'} {out_clip}")
    (output_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print("[done] manifest:",output_dir/"manifest.json")

if __name__=="__main__": main()
