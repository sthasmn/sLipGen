# Use the official NVIDIA CUDA base image. This includes the drivers and the toolkit.
# Using a specific version ensures reproducibility.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, including Python, pip, and ffmpeg
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install Python dependencies from requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download the pre-trained models during the build process
# This ensures they are available when the container runs
RUN pip install --no-cache-dir gdown
RUN python -c "from model.faceDetector.s3fd import S3FD; S3FD(device='cpu')"
# Note: The main ASD model weight ('finetuning_TalkSet.model') is expected to be in the 'weight/' directory.
# The Whisper model will be downloaded on first use by the library itself.

# Set the entrypoint to be an interactive bash shell
# This allows the user to run the optimized_runner.py script with their own video paths
CMD ["/bin/bash"]
