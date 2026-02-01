import os
import uuid
import subprocess

import pdb

FRAMERATE = 30
VIDEO_SHAPE = (640, 480)
VIDEO_DEVICE = "/dev/video0"
VIDEO_ENCODER = "libx264"
VIDEO_ENCODING_SPEED = "medium"
AUDIO_DEVICE = "default"
CONSTANT_RATE_FACTOR = 23
AUDIO_ENCODER = "aac"
RECORD_TIME = 300 

NUM_FILES = 10
S3_BUCKET_NAME = "streamswitch-streams"

command_options = [
    ("-f", "v4l2"),
    ("-framerate", f"{FRAMERATE}"),
    ("-video_size", f"{VIDEO_SHAPE[0]}x{VIDEO_SHAPE[1]}"),
    ("-i", f"{VIDEO_DEVICE}"),
    ("-f", "alsa"),
    ("-i", f"{AUDIO_DEVICE}"),
    ("-c:v", f"{VIDEO_ENCODER}"),
    ("-preset", f"{VIDEO_ENCODING_SPEED}"),
    ("-crf", f"{CONSTANT_RATE_FACTOR}"),
    ("-c:a", f"{AUDIO_ENCODER}"),
    ("-b:a", "128k"),
    ("-t", f"{RECORD_TIME}"),
    ("-f", "mp4")
]

aws_command = ["aws", "s3", "mv"]

collection_command = ["ffmpeg"]
for option, value in command_options:
    collection_command.append(option)
    collection_command.append(value)


try:
    while True:
        doc_id = uuid.uuid4()
        collection_command.append(f"{doc_id}.mp4")
        cmd = " ".join(collection_command)
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
        if result.returncode == 0:
            print(f"\033[1;32mFILE {doc_id} video recording complete \033[0;0m")
        else:
            print(f"\033[1;31mFILE {doc_id} video recording fail \033[0;0m")
            collection_command.pop()
            aws_command.pop()
            aws_command.pop()
            continue

        new_s3_object = f"s3://{S3_BUCKET_NAME}/{doc_id}.mp4"
        aws_command.append(f"{doc_id}.mp4")
        aws_command.append(new_s3_object)
        cmd = " ".join(aws_command)
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
        if result.returncode == 0:
            print(f"\033[1;32mFILE {doc_id} video upload complete \033[0;0m")
        else:
            print(f"\033[1;31mFILE {doc_id} video upload fail \033[0;0m")

        collection_command.pop()
        aws_command.pop()
        aws_command.pop()
except KeyboardInterrupt as e:
    print("PROGRAM ABORTED")
    sys.exit(0)
