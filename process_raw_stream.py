import sys, os
import shutil
import cv2
import argparse
import moviepy as mp
from pathlib import Path

cli_parser = argparse.ArgumentParser(prog="Raw Stream Processor")
cli_parser.add_argument(
    "video_file",
    metavar="/path/to/file",
    type=str
)

cli_args = vars(cli_parser.parse_args())
video_file = Path(cli_args["video_file"])

if not video_file.exists() or not video_file.is_file():
    print(f"{video_file} is invalid!")
    sys.exit(1)

processed_video_dir = video_file.cwd().joinpath(video_file.name.split(".")[0])
processed_video_dir.mkdir(exist_ok=True)
clip = mp.VideoFileClip(video_file)

while True:
    try:
        start_time, end_time = input("Enter time range: ").split()
        clipped_video = clip.subclipped(start_time, end_time)
        clipped_video.write_videofile("clipped_video.mp4", codec="libx264", audio_codec="aac")
        break
    except ValueError:
        print("Invalid time range input!")

base_path = Path(os.getcwd())
image_frames = base_path.joinpath("video_frames")
if image_frames.exists():
    shutil.rmtree(image_frames)
image_frames.mkdir(exist_ok=True)

cap = cv2.VideoCapture(base_path.joinpath("clipped_video.mp4"))

if not cap.isOpened():
    print(f"Could not open {video_path}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_num = 0
while True:

    ret, frame = cap.read()

    if not ret:
        print("Finished reading frames.")
        break

    jpeg_file = image_frames.joinpath(image_frames, f"frame_{frame_num}.jpg")
    cv2.imwrite(jpeg_file, frame)

    if frame_num % 100 == 0: 
        print(f"Frame {frame_num} saved of {total_frames}.")
      
    frame_num += 1

cap.release()