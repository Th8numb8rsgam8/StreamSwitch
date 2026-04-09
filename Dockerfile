# Start from official Debian image
# FROM debian:bookworm-slim
FROM tensorflow/tensorflow:2.21.0-gpu

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    # wget \
    # bzip2 \
    # curl \
    # cuda-cudart-12-5 \
    # cuda-compat-12-5 \
    cudnn9-cuda-12 \
    # libcudnn9-dev-cuda-12 \
    # gcc \
    # g++ \
    # libssl-dev \
    # ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
# RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
#     && rm Miniconda3-latest-Linux-x86_64.sh
# ENV PATH="/opt/conda/bin:${PATH}"

# Create and activate a conda environment
# RUN conda tos accept
# RUN conda create --name streamswitch python=3.11.13 -y
# ENV PATH="/opt/conda/envs/streamswitch/bin:${PATH}"

# Install SageMaker Training Toolkit (required for integration)
RUN pip install sagemaker-training==5.1.1
RUN pip install --upgrade --force-reinstall cryptography pyOpenSSL

# Install Python packages into Anaconda environment
COPY training_job_requirements.txt /tmp/training_job_requirements.txt
RUN pip install --no-cache-dir -r /tmp/training_job_requirements.txt

# Set paths
# ENV TF_CPP_MAX_VLOG_LEVEL=3
ENV CUDNN_PATH=/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Define SageMaker environment variables
WORKDIR /opt/ml/code
ENV SAGEMAKER_PROGRAM streamswitch.py