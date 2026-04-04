import os, sys
import shutil
import cv2
import json
import subprocess
from pathlib import Path
import scipy.io.wavfile as wavfile

import pdb

base_path = Path(os.getcwd())
image_frames = base_path.joinpath("video_frames")
training_data = base_path.parent.joinpath("training_data")

result = subprocess.run("aws s3 ls s3://streamswitch-training-data", shell=True, capture_output=True, text=True)
video_hashes = result.stdout.replace(" ", "").replace("PRE", "").replace("/", "").split("\n")
video_hashes.pop()

for video_hash in video_hashes:
    metadata_file = training_data.joinpath(video_hash, "metadata.json")
    # result = subprocess.run(f"aws s3 cp s3://streamswitch-training-data/{video_hash}/video.mp4 .", shell=True)
    # if result.returncode == 0:
    #     print(f"VIDEO {video_hash} DOWNLOAD SUCCESS")

    # audio_sample_rate = subprocess.run(
    #     f'ffprobe {base_path.joinpath("video.mp4")} -show_entries stream=sample_rate -select_streams a:0 -of compact=p=0:nk=1 -v 0',
    #     shell=True, capture_output=True, text=True)

    result = subprocess.run(f"aws s3 cp s3://streamswitch-training-data/{video_hash}/audio.wav .", shell=True)
    if result.returncode == 0:
        print(f"VIDEO {video_hash} DOWNLOAD SUCCESS")
    sr, wav_audio = wavfile.read(base_path.joinpath("audio.wav"))

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
        # metadata["audio_sample_rate"] = int(audio_sample_rate.stdout)
        metadata["total_audio_frames"] = wav_audio.shape[0]
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # if image_frames.exists():
    #     shutil.rmtree(image_frames)
    # image_frames.mkdir(exist_ok=True)

    # cap = cv2.VideoCapture(base_path.joinpath("video.mp4"))
    # if not cap.isOpened():
    #     print(f"Could not open processed video!")
    #     sys.exit(1)

    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # frame_num = 0
    # while True:

    #     ret, frame = cap.read()

    #     if not ret:
    #         print("Finished reading frames.")
    #         break

    #     jpeg_file = image_frames.joinpath(f"frame_{frame_num}.jpg")
    #     cv2.imwrite(jpeg_file, frame)

    #     if frame_num % 100 == 0: 
    #         print(f"Frame {frame_num} saved of {total_frames}.")
    #       
    #     frame_num += 1
    # cap.release()

    # result = subprocess.run(f"aws s3 mv {image_frames} s3://streamswitch-training-data/{video_hash}/video_frames --recursive", shell=True)
    # if result.returncode == 0:
    #     print(f"{video_hash} IMAGE FRAME UPLOAD SUCCESS!")

    # result = subprocess.run(f'ffmpeg -i {base_path.joinpath("video.mp4")} -vn -acodec pcm_s16le -ar {int(audio_sample_rate.stdout)} -ac 2 audio.wav', shell=True)
    # result = subprocess.run(f'aws s3 mv {base_path.joinpath("audio.wav")} s3://streamswitch-training-data/{video_hash}/audio.wav', shell=True)
    # if result.returncode == 0:
    #     print(f"{video_hash} AUDIO WAV FILE UPLOAD SUCCESS WITH SAMPLE RATE#{int(audio_sample_rate.stdout)}!")

    result = subprocess.run(f'aws s3 cp {metadata_file} s3://streamswitch-training-data/{video_hash}/metadata.json', shell=True)
    if result.returncode == 0:
        print(f"{video_hash} METADATA FILE UPLOAD SUCCESS!")

    os.remove(base_path.joinpath("audio.wav"))