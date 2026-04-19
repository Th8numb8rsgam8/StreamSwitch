#!/bin/bash

ACCOUNT_ID=$1
IMAGE_NAME=$2
REPOSITORY_NAME=$3

aws ecr get-login-password | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# aws sagemaker update-domain --domain-id d-jtha32prud4l --domain-settings-for-update '{"DockerSettings": {"EnableDockerAccess": "ENABLED"}}'

docker build --network sagemaker -t ${IMAGE_NAME} .
docker tag ${IMAGE_NAME}:latest ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${REPOSITORY_NAME}
docker push ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${REPOSITORY_NAME}

# docker run -it --name my_dev_container <image_uri> /bin/bash