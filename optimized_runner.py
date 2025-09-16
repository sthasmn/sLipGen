import sys
import os
import subprocess
import json
from pathlib import Path
import torch
import whisper
import multiprocessing as mp
from typing import List

# Import processing functions from the utility file
from pipeline_utils import run_video_preprocessing, run_vision_pipeline, run_transcription, run_final_muxing, \
    PipelineTask


# --- Configuration ---
class Config:
    """A single configuration object for the entire pipeline"""
    # --- Device Assignment ---
    # Assign heavy tasks to different GPUs for parallel processing
    VISION_DEVICE: str = "cuda:0"
    WHISPER_DEVICE: str = "cuda:1"

    # --- Whisper Model ---
    WHISPER_MODEL_SIZE: str = "large-v3-turbo"
    WHISPER_LANGUAGE: str = "ja"

    # --- ASD Model ---
    PRETRAIN_MODEL_PATH: str = "weight/finetuning_TalkSet.model"

    # --- Preprocessing and Detection Parameters ---
    FACEDET_SCALE: float = 0.25
    MIN_TRACK: int = 10
    NUM_FAILED_DET: int = 10
    MIN_FACE_SIZE: int = 1
    CROP_SCALE: float = 0.40
    N_DATA_LOADER_THREAD: int = 10  # For ffmpeg

    # --- Speaker Selection ---
    MIN_SPEAKER_SCORE_THRESHOLD: float = 0.5

    # --- Worker Configuration ---
    # Number of CPU workers for the initial video decoding
    NUM_PREPROCESS_WORKERS: int = max(1, mp.cpu_count() // 2)


# =====================================================================================
#  Worker Functions for Each Pipeline Stage
# =====================================================================================

def preprocess_worker(
        task_queue: mp.Queue,
        out_queue: mp.Queue,
        config: Config
):
    """
    Stage 1 (CPU Pool): Takes video paths, extracts frames and audio, and passes a task object to the next stage.
    """
    while True:
        video_path_str = task_queue.get()
        if video_path_str is None:
            break

        video_path = Path(video_path_str)
        print(f"[Pre-process] Starting: {video_path.name}")
        try:
            task = run_video_preprocessing(video_path, config)
            if task:
                out_queue.put(task)
                print(f"[Pre-process] Finished: {video_path.name}")
        except Exception as e:
            print(f"!!! [Pre-process] Error processing {video_path.name}: {e}")

    # Signal that this worker is done
    out_queue.put(None)


def vision_worker(
        in_queue: mp.Queue,
        out_queue: mp.Queue,
        config: Config
):
    """
    Stage 2 (GPU 0): Runs the entire vision pipeline (scenes, faces, tracks, ASD scores).
    """
    print(f"[Vision] Loading models on {config.VISION_DEVICE}...")
    # This function now initializes models inside
    print(f"[Vision] Models loaded. Ready for processing.")

    while True:
        task = in_queue.get()
        if task is None:
            break

        print(f"[Vision] Starting vision pipeline for: {task.video_path.name}")
        try:
            processed_task = run_vision_pipeline(task, config)
            out_queue.put(processed_task)
            print(f"[Vision] Finished vision pipeline for: {task.video_path.name}")
        except Exception as e:
            print(f"!!! [Vision] Error processing {task.video_path.name}: {e}")
            import traceback
            traceback.print_exc()

    out_queue.put(None)


def whisper_worker(
        in_queue: mp.Queue,
        out_queue: mp.Queue,
        config: Config
):
    """
    Stage 3 (GPU 1): Runs Whisper transcription on the audio.
    """
    print(f"[Whisper] Loading model '{config.WHISPER_MODEL_SIZE}' on {config.WHISPER_DEVICE}...")
    model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=config.WHISPER_DEVICE)
    print("[Whisper] Model loaded. Ready for transcription.")

    while True:
        task = in_queue.get()
        if task is None:
            break

        print(f"[Whisper] Transcribing: {task.video_path.name}")
        try:
            transcribed_task = run_transcription(task, model, config)
            out_queue.put(transcribed_task)
            print(f"[Whisper] Finished transcription for: {task.video_path.name}")
        except Exception as e:
            print(f"!!! [Whisper] Error transcribing {task.video_path.name}: {e}")

    out_queue.put(None)


def mux_worker(in_queue: mp.Queue, config: Config):
    """
    Stage 4 (CPU): Takes the final task object with all data and creates the output clips.
    """
    while True:
        task = in_queue.get()
        if task is None:
            break

        print(f"[Muxer] Creating final clips for: {task.video_path.name}")
        try:
            run_final_muxing(task, config)
            print(f"[Muxer] Finished: {task.video_path.name}")
        except Exception as e:
            print(f"!!! [Muxer] Error finalizing {task.video_path.name}: {e}")
            import traceback
            traceback.print_exc()


# =====================================================================================
#  Main Orchestrator
# =====================================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python optimized_runner.py /mnt/e/JCOM/RAW_Data/ふわっと欣様/")
        sys.exit(1)

    config = Config()

    # --- Find Video Files ---
    input_path = Path(sys.argv[1])
    if input_path.is_dir():
        video_files = [str(f) for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MXF", "*.mxf") for f in
                       input_path.glob(ext)]
    elif input_path.is_file():
        video_files = [str(input_path)]
    else:
        print(f"Error: Path '{input_path}' is not a valid file or directory.")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s) to process.")
    if not video_files:
        return

    # --- Setup Multiprocessing ---
    mp.set_start_method("spawn", force=True)

    # Create queues to connect the pipeline stages
    preprocess_to_vision_queue = mp.Queue()
    vision_to_whisper_queue = mp.Queue()
    whisper_to_mux_queue = mp.Queue()

    # --- Create Worker Processes ---
    # Stage 1: Pre-processing (CPU Pool)
    preprocess_task_queue = mp.Queue(len(video_files))
    for vf in video_files:
        preprocess_task_queue.put(vf)

    preprocess_procs = [
        mp.Process(target=preprocess_worker, args=(preprocess_task_queue, preprocess_to_vision_queue, config))
        for _ in range(config.NUM_PREPROCESS_WORKERS)
    ]

    # Stage 2: Vision Pipeline (GPU 0)
    vision_proc = mp.Process(target=vision_worker, args=(preprocess_to_vision_queue, vision_to_whisper_queue, config))

    # Stage 3: Transcription (GPU 1)
    whisper_proc = mp.Process(target=whisper_worker, args=(vision_to_whisper_queue, whisper_to_mux_queue, config))

    # Stage 4: Final Muxing (CPU)
    mux_proc = mp.Process(target=mux_worker, args=(whisper_to_mux_queue, config))

    # --- Start and Manage the Pipeline ---
    all_procs = preprocess_procs + [vision_proc, whisper_proc, mux_proc]
    for p in all_procs:
        p.start()

    # Send termination signals to the pre-process workers
    for _ in range(config.NUM_PREPROCESS_WORKERS):
        preprocess_task_queue.put(None)

    # Wait for the pipeline to complete
    active_preprocess_workers = config.NUM_PREPROCESS_WORKERS
    while active_preprocess_workers > 0:
        task = preprocess_to_vision_queue.get()
        if task is None:
            active_preprocess_workers -= 1
        else:
            vision_to_whisper_queue.put(task)
    vision_to_whisper_queue.put(None)  # Signal vision worker to terminate

    while True:
        task = vision_to_whisper_queue.get()
        if task is None:
            break
        else:
            whisper_to_mux_queue.put(task)
    whisper_to_mux_queue.put(None)  # Signal whisper worker to terminate

    mux_proc.join()  # Wait for the final stage to finish

    # Ensure all processes have terminated
    for p in all_procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()

