import os
import sys
import json
from pathlib import Path
import librosa
import scipy.io.wavfile as wavfile
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


if __name__ == "__main__":

    print(f"PYTHON VERSION: {sys.version}")
    print("LET THE TRAINING BEGIN!")
    training_data_root_dir = Path(os.environ["SM_CHANNEL_TRAIN"])
    video_hash = "0035e03a-2365-4bc7-920e-630050a93e2e"
    dirs = [d for d in training_data_root_dir.joinpath("streamswitch_fsx").iterdir()]
    print(f"NUM VIDEOS: {len(dirs)}")
    with open(training_data_root_dir.joinpath("streamswitch_fsx", video_hash, "metadata.json")) as f:
        metadata = json.load(f)
        print(metadata)