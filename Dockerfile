# Use the official NVIDIA CUDA base image. This includes the drivers and the toolkit.
# Using a specific version ensures reproducibility.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Set Python to run in unbuffered mode, which is useful for seeing logs in real-time
ENV PYTHONUNBUFFERED=1

# Install system dependencies, including Python, pip, ffmpeg, and an SSH server
RUN apt-get update && apt-get install -y \
    openssh-server \
    sudo \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# --- SSH Server and User Setup ---
# Create a non-root user 'lipgen' with password 'lipgen' and add to sudo group
RUN useradd -m -s /bin/bash lipgen && \
    echo "lipgen:lipgen" | chpasswd && \
    adduser lipgen sudo

# Configure the SSH server to allow password authentication
RUN mkdir /var/run/sshd
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
RUN sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Expose the SSH port
EXPOSE 22

# --- Project Setup ---
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first
# This leverages Docker's layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the working directory
COPY . .

# Set ownership of the app directory to the new user
RUN chown -R lipgen:lipgen /app

# The final command starts the SSH daemon by default, keeping the container running.
CMD ["/usr/sbin/sshd", "-D"]

