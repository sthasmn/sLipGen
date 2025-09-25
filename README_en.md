# A System for Automated Active Speaker Detection and Temporal Video Segmentation

## Project Goal

The primary objective of this system is to facilitate the creation of a large-scale, Japanese-language lip-reading dataset, analogous to established English-language datasets such as LRS2 and LRS3. The pipeline is specifically engineered to process "in-the-wild" video content, with a particular focus on Japanese television programming, to generate a corpus of temporally segmented video clips suitable for training advanced lip-reading models.

## Core System Capabilities

* **Active Speaker Detection**: The system integrates a sophisticated audio-visual model to ascertain the identity of the enunciating individual at any given temporal point within the video. The model's performance is reported to be of a high degree of accuracy.
* **Automated Speech Recognition**: Speech content is transcribed into text through the implementation of OpenAI's Whisper model. This process yields not only the textual transcription but also precise, word-level temporal metadata.
* **Scene-Aware Facial Tracking**: The methodology incorporates a scene detection algorithm, which serves to constrain the facial tracking process. This prevents the erroneous concatenation of facial identities across cinematic cuts.
* **Utterance-Level Video Segmentation**: The system programmatically correlates identified speakers with transcribed sentences, thereby enabling the automated generation of discrete video segments and corresponding textual transcripts for each distinct utterance.
* **Intermediate State Caching**: To facilitate expedited re-processing, the system caches the computationally intensive results of the vision analysis pipeline. Subsequent executions can leverage these cached artifacts, significantly reducing processing time when only downstream parameters are modified.
* **Structured Data Output**: All generated artifacts are organized within a structured directory hierarchy. A `manifest.json` file is produced to catalogue the resultant video clips and their associated metadata.

## System Architecture and Workflow

The operational workflow of the system is delineated by a sequential, multi-stage pipeline, the architecture of which is illustrated below.

```
graph TD
    A[Input Video] --> B{Stage 1: Pre-processing};
    B --> C[Extract All Frames];
    B --> D[Extract & Resample Audio];
    
    C & D --> E{Stage 2: Vision Analysis};
    E --> F[Detect Scene Changes];
    E --> G[Detect Faces in Every Frame];
    F & G --> H[Track Faces Within Scenes];
    
    H & D --> I{Stage 3: Speaker Scoring (ASD)};
    I --> J[For each Face Track, Crop Video & Audio];
    I --> K[Score Audio-Visual Sync with ASD Model];
    
    D --> L{Stage 4: Transcription};
    L --> M[Transcribe Full Audio with Whisper];
    
    K & M --> N{Stage 5: Final Assembly};
    N --> O[For each Transcribed Sentence...];
    O --> P{Find Best-Matching Face Track based on ASD Score};
    P --> Q[Generate Final .mp4 Clip & .txt File];
    Q --> R[Write manifest.json];
    R --> S[End];
```

## System Deployment and Configuration

Two methods for system deployment are provided: using Docker for a containerized environment (recommended) or performing a manual installation on a host machine.

### Option A: Using Docker (Recommended)

This method provides a self-contained, reproducible environment with all dependencies pre-installed. You can run it in two ways.

#### **Initial Setup (Do This Once)**
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/sthasmn/sLipGen.git](https://github.com/sthasmn/sLipGen.git)
    cd sLipGen
    ```

2.  **Build the Docker Image:**
    From the root of the project directory, execute the following command.
    ```bash
    docker build -t slipgen .
    ```

#### **Mode 1: Direct Processing (One-Off Task)**
This is the simplest way to process a batch of videos. The container starts, runs the script, and stops when finished.
```bash
# Replace /path/to/your/local/videos with the actual path on your machine
docker run --gpus all --rm -it \\
  -v /path/to/your/local/videos:/videos \\
  slipgen python main.py /videos/
```
To use the `skip-asd` flag, simply append it to the command:
```bash
docker run --gpus all --rm -it \\
  -v /path/to/your/local/videos:/videos \\
  slipgen python main.py /videos/ skip-asd
```

#### **Mode 2: Persistent Container (for SSH & Development)**
This mode is useful for debugging or running other scripts. The container runs continuously in the background as an SSH server.

1.  **Start the Persistent Container:**
    ```bash
    # Replace /path/to/your/local/videos with the actual path on your machine
    docker run --gpus all -d --name slipgen-dev \\
      -v /path/to/your/local/videos:/videos \\
      -p 127.0.0.1:2222:22 \\
      slipgen
    ```

2.  **Connect via SSH:**
    * **User:** `lipgen`
    * **Password:** `lipgen`
    ```bash
    ssh lipgen@localhost -p 2222
    ```

3.  **Run Commands Inside:**
    ```bash
    # Example: Run the main processing script
    python main.py /videos/
    ```

4.  **Stopping the Container:**
    ```bash
    docker stop slipgen-dev
    ```

### Option B: Manual Installation

#### **Prerequisites**
* An Ubuntu-based operating system is recommended.
* An NVIDIA GPU with a minimum of 16GB of VRAM.
* NVIDIA drivers and the CUDA Toolkit.
* `conda` or `miniconda`.
* `ffmpeg` installed and in the system's PATH.

#### **Environment Configuration**
1.  **Repository Acquisition:**
    ```bash
    git clone [https://github.com/sthasmn/sLipGen.git](https://github.com/sthasmn/sLipGen.git)
    cd sLipGen
    ```

2.  **Environment Initialization:**
    ```bash
    conda create -n sLipGen python=3.10 -y
    conda activate sLipGen
    ```

3.  **Installation of Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Operational Procedures (Manual Installation)

### 1. Parameter Configuration

Modify parameters within the `Config` class in `main.py`.

* `DEVICE`: Target CUDA device (e.g., `"cuda:0"`).
* `WHISPER_MODEL_SIZE`: Whisper model variant.
* `MIN_SPEAKER_SCORE_THRESHOLD`: Minimum ASD confidence score.

### 2. Pipeline Execution

Run the main script from your activated conda environment.
```bash
python main.py /path/to/your/videos/
```
To use the `skip-asd` flag:
```bash
python main.py /path/to/your/videos/ skip-asd
```

## Output Artifacts and Directory Structure

For an input file (`my_video.mp4`), a corresponding directory (`my_video/`) is generated with the following structure:

```
my_video/
├── pyavi/
│   └── audio.wav         # Extracted master audio file
├── pycrop/               # Temporary cropped clips for ASD scoring
├── pyframes/             # All extracted video frames as JPGs
├── pywork/
│   ├── faces.pckl        # Face detection data
│   ├── scene.pckl        # Scene change data
│   ├── scores.pckl       # ASD scores for each track
│   └── tracks.pckl       # Face track data
└── output/
    ├── seg_0001_track_005.mp4
    ├── seg_0001_track_005.txt
    ├── ...
    └── manifest.json     # JSON summary of all generated clips
```

## Acknowledgements

This work is predicated on the capabilities of several foundational open-source projects:
* **TalkNet**: The core Active Speaker Detection model architecture.
* **S3FD**: The facial detection model utilized herein.
* **Whisper**: The automated speech recognition model developed by OpenAI.
* **PySceneDetect**: The library employed for scene boundary detection.