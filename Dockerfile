# Use the official NVIDIA CUDA base image. This includes the drivers and the toolkit.
# Using a specific version ensures reproducibility.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

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

# Create a non-root user for development with a default password 'developer'
RUN useradd -m -s /bin/bash developer && \
    echo "lipgen:lipgen" | chpasswd && \
    adduser developer sudo

# Configure the SSH server
RUN mkdir /var/run/sshd
# --- EDIT: Re-enable password authentication for internal network use ---
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
RUN sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config


# Expose the SSH port
EXPOSE 22

# --- Project Setup ---

# Set the working directory inside the container. You will clone your repo here.
WORKDIR /app

# The user will install Python dependencies manually after cloning the repo.
# The final command starts the SSH daemon and keeps the container running.
CMD ["/usr/sbin/sshd", "-D"]

