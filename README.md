# High-Performance Active Speaker Detection & Clip Generation Pipeline

This project provides a complete, high-performance pipeline for processing videos to identify active speakers and automatically generate precisely timed, sentence-level video clips of their faces. It leverages a state-of-the-art audio-visual deep learning model and is optimized for both single and multi-GPU systems to achieve maximum throughput on large batches of "in-the-wild" videos.

The core of the project is the `optimized_runner.py` script, which implements a parallel, multi-stage processing assembly line.

## Core Features

* **Accurate Active Speaker Detection (ASD)**: Utilizes a powerful audio-visual model to determine who is speaking at any given moment with high accuracy.

* **Automated Transcription**: Employs OpenAI's Whisper model for precise, word-level timestamped transcriptions.

* **Sentence-Level Clip Generation**: Automatically correlates speakers with transcribed sentences to generate individual video clips for each line of dialogue.

* **Flexible GPU Architecture**: Natively supports both single and multi-GPU setups through simple configuration.

* **Parallel Processing Pipeline**: Implements a multi-stage, queue-based system that processes multiple videos simultaneously, maximizing CPU and GPU utilization.

* **Structured Output**: Generates a clean directory structure for each processed video, including a `manifest.json` file that catalogues all generated clips.

## System Architecture

The pipeline consists of four main stages that run concurrently, coordinated by the `optimized_runner.py` script. This design eliminates bottlenecks and ensures that both CPU and GPU resources are kept busy.

1. **[CPU Pool] Pre-processing**: Decodes videos into frames and extracts audio.

2. **[GPU] Transcription**: A dedicated worker runs Whisper to generate transcripts with word-level timestamps.

3. **[GPU] Vision & Scoring**: A second worker runs the entire vision pipeline: scene detection, face tracking, and ASD scoring.

4. **[CPU] Muxing & Output**: A final worker correlates the results, generates the final clips using `ffmpeg`, and writes the `manifest.json`.

## Setup and Installation

### Option A: Using Docker (Recommended)

This is the easiest and most reproducible method. Ensure you have Docker and the NVIDIA Container Toolkit installed.

1. **Build the Docker Image**:

   ```
   docker build -t lipgen-asd .
   ```

2. **Run the Container**:
   Mount your project directory and a directory containing your videos into the container.

   ```
   docker run --gpus all -it --rm \
     -v /path/to/your/project:/app \
     -v /path/to/your/videos:/videos \
     lipgen-asd
   ```

### Option B: Manual Installation

1. **Prerequisites**:

   * An Ubuntu-based system (tested on WSL2 with Ubuntu 22.04).

   * One or more NVIDIA GPUs with at least 12GB of VRAM.

   * NVIDIA drivers installed on the host system.

   * `conda` or `miniconda` installed.

2. **Environment Setup**:

   ```
   git clone <your-repo-url>
   cd <your-repo-name>
   
   # Create a conda environment
   conda create -n sLipGen python=3.10 -y
   conda activate sLipGen
   
   # Install PyTorch with CUDA support
   # Check [https://pytorch.org/](https://pytorch.org/) for the command matching your CUDA version
   pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
   
   # Install remaining dependencies from requirements.txt
   pip install -r requirements.txt
   ```

## Usage Guide

### 1. Configuration (Crucial Step)

Before running, open `optimized_runner.py` and configure the GPU settings within the `Config` class based on your hardware.

**For a Dual-GPU setup (Maximum Performance):**

```
class Config:
    WHISPER_DEVICE: str = "cuda:0"  # Assign Whisper to your first GPU
    VISION_DEVICE: str = "cuda:1"   # Assign Vision models to your second GPU
    # ... other settings
```

**For a Single-GPU setup:**

```
class Config:
    WHISPER_DEVICE: str = "cuda:0"  # Assign Whisper to your only GPU
    VISION_DEVICE: str = "cuda:0"   # Assign Vision models to the SAME GPU
    # ... other settings
```

### 2. Running the Pipeline

Place all your input videos into a single directory. Then, from within your Docker container or activated conda environment, run the script:

```
python optimized_runner.py /path/to/your/videos/
```

### 3. Understanding the Output

For each input video (e.g., `my_video.mp4`), a corresponding output directory (`my_video/`) will be created containing intermediate files and the final clips in an `output/` subdirectory. The `manifest.json` file provides a detailed summary of each generated clip.

## Acknowledgements

This work is built upon several incredible open-source projects:

* **TalkNet**: The core ASD model architecture.

* **S3FD**: The face detection model.

* **Whisper**: The transcription model from OpenAI.

* **PySceneDetect**: Used for robust scene detection.