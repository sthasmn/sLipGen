"""
FPS Detection Comparison Tool

This script iterates through a directory of video files and compares the FPS
(frames per second) value detected by two different methods:
1.  OpenCV (`cv2.VideoCapture`): The method used in the original script, which
    can be unreliable and defaults to 25 FPS if metadata is not found.
2.  ffprobe: A robust command-line tool from the FFmpeg suite, considered the
    industry standard for reading media file metadata.

This tool helps diagnose issues where an incorrect FPS value might be causing
desynchronization or errors in video processing pipelines.
"""

import sys
import cv2
import subprocess
import math
from pathlib import Path

def get_fps_with_opencv(video_path: Path) -> float:
    """
    Gets the FPS using the original script's OpenCV method.
    Defaults to 25.0 if detection fails.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        # This is the exact logic from the original script
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return float(fps)
    except Exception as e:
        print(f"  [OpenCV Error] Could not process {video_path.name}: {e}")
        return 0.0

def get_fps_with_ffprobe(video_path: Path) -> float:
    """
    Gets the FPS using the highly reliable ffprobe command.
    Returns 0.0 if ffprobe fails.
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        # ffprobe can return a fraction (e.g., '30000/1001') or a number
        if '/' in output:
            num, den = map(int, output.split('/'))
            return num / den if den != 0 else 0.0
        else:
            return float(output)
    except FileNotFoundError:
        print("\n[ERROR] ffprobe command not found.", file=sys.stderr)
        print("Please ensure ffmpeg is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError:
        print(f"  [ffprobe Error] Could not get FPS for {video_path.name}.", file=sys.stderr)
        return 0.0

def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        video_dir = Path("/mnt/e/JCOM/RAW_Data/ふわっと欣様")
        print("Usage: python check_fps.py <path_to_video_directory>")
        #sys.exit(1)

    else :
        video_dir = Path(sys.argv[1])
    if not video_dir.is_dir():
        print(f"Error: Provided path '{video_dir}' is not a valid directory.")
        sys.exit(1)

    print(f"Scanning for videos in: {video_dir}\n")
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MXF", "*.mxf")
    video_files = sorted([f for ext in video_extensions for f in video_dir.glob(ext)])

    if not video_files:
        print("No video files found.")
        return

    results = []
    max_len = max(len(p.name) for p in video_files) + 2

    print("Processing files...")
    for video_path in video_files:
        print(f"- {video_path.name}")
        opencv_fps = get_fps_with_opencv(video_path)
        ffprobe_fps = get_fps_with_ffprobe(video_path)
        results.append({
            "name": video_path.name,
            "opencv": opencv_fps,
            "ffprobe": ffprobe_fps,
        })

    print("\n--- FPS Detection Comparison Report ---\n")
    header = f"{'Filename':<{max_len}} | {'OpenCV (Original)':<20} | {'ffprobe (Correct)':<20} | {'Match?':<10}"
    print(header)
    print('-' * len(header))

    mismatch_count = 0
    for res in results:
        name = res['name']
        o_fps = res['opencv']
        f_fps = res['ffprobe']

        # Use math.isclose for safe floating point comparison
        is_match = math.isclose(o_fps, f_fps)
        status = "✅ MATCH" if is_match else "❌ MISMATCH"
        if not is_match:
            mismatch_count += 1

        print(f"{name:<{max_len}} | {o_fps:<20.2f} | {f_fps:<20.2f} | {status:<10}")

    print("\n--- Summary ---")
    print(f"Found {mismatch_count} mismatches out of {len(video_files)} files.")
    if mismatch_count > 0:
        print("Recommendation: Update your main script to use the ffprobe method for accurate FPS detection.")
    else:
        print("All detected FPS values match. The original method appears to be working for this set of files.")

if __name__ == "__main__":
    main()
