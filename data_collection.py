import os
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
RECORD_TIME = 60

NUM_FILES = 10

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

collection_command = ["ffmpeg"]
for option, value in command_options:
    collection_command.append(option)
    collection_command.append(value)


for i in range(NUM_FILES):
    collection_command.append(f"output_{i}.mp4")
    cmd = " ".join(collection_command)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    if result.returncode == 0:
        print(f"FILE {i} video recording complete")
    else:
        print(f"FILE {i} video recording fail")
    collection_command.pop()
