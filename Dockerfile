# Start from official Debian image
FROM debian:bookworm-slim

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    gcc \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Create and activate a conda environment
RUN conda tos accept
RUN conda create --name streamswitch python=3.11.13 -y
ENV PATH="/opt/conda/envs/streamswitch/bin:${PATH}"

# Install SageMaker Training Toolkit (required for integration)
RUN pip install sagemaker-training==4.7.0

# Install Python packages into Anaconda environment
COPY training_job_requirements.txt /tmp/training_job_requirements.txt
RUN pip install --no-cache-dir -r /tmp/training_job_requirements.txt

# Define SageMaker environment variables
WORKDIR /opt/ml/code
ENV SAGEMAKER_PROGRAM script.py