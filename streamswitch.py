import os
import math
import cv2
import librosa
from spafe.features.gfcc import gfcc
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from pydub import AudioSegment
from pathlib import Path

import pdb

NUM_CLASSES = 2

base_path = Path(os.getcwd())

# image_frames = base_path.joinpath("image_frames")
# image_frames.mkdir(exist_ok=True)
# 
# cap = cv2.VideoCapture(base_path.joinpath("Football_Clip.mp4"))
# 
# if not cap.isOpened():
    # print(f"Could not open {video_path}")
    # sys.exit(1)
# 
# frame_num = 0
# while True:
# 
    # ret, frame = cap.read()
# 
    # if not ret:
        # break
# 
    # jpeg_file = image_frames.joinpath(image_frames, f"frame_{frame_num}.jpg")
    # cv2.imwrite(jpeg_file, frame)
# 
    # if frame_num % 100 == 0: 
        # print(f"Frame {frame_num} saved.")
    # 
    # frame_num += 1
# 
# cap.release()

################################################################################

# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# 
# labels = np.random.randint(0, 1, size=total_frames)

# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
# 
# ret, frame = cap.read()

# cv2.imshow(f"Frame {frame_number}", frame)
# cv2.waitKey(0)

class FrameGenerator:

    def __init__(self, video_paths, video_group_size, target_shape=(224, 224)):

        self._video_paths = video_paths
        self._video_group_size = video_group_size
        self._target_shape = target_shape
        self._num_cepstral_coeffs = 13
        # nrows = ((a.size - stride_length) // stride_step) + 1

    def _get_frames(self, video_path):

        video = cv2.VideoCapture(video_path)
        audio = AudioSegment.from_file(video_path, format="mp4")

        if not video.isOpened():
            print(f"Could not open {video_path}")
            return
            # sys.exit(1)

        total_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = video.get(cv2.CAP_PROP_FPS)
 
        num_groups, unused_video_frames = divmod(total_video_frames, self._video_group_size)
        audio_per_video, unused_audio_frames = divmod(int(audio.frame_count() / audio.channels), total_video_frames)
        audio_group_size = audio_per_video * self._video_group_size

        print(f"{video_path}: UNUSED VIDEO FRAMES {unused_video_frames}, UNUSED AUDIO FRAMES {unused_audio_frames}.")

        audio_idx = 0
        for _ in range(num_groups):
            video_frame_group = self._get_video_group(video)
            mfcc_coeffs, gfcc_coeffs = self._get_audio_group(audio, audio_idx, audio_idx + audio_group_size)
            audio_idx += audio_group_size

            pdb.set_trace()

            yield video_frame_group
     
        video.release()
    
    def _get_video_group(self, video):

        frame_grp = []
        for _ in range(self._video_group_size):
            ret, frame = video.read()
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            frame = tf.image.resize(frame, self._target_shape)
            frame_grp.append(frame)
        return tf.stack(frame_grp, axis=0)
    
    def _get_audio_group(self, audio, start_idx, end_idx):

        result = audio.get_sample_slice(start_idx, end_idx).get_array_of_samples()
        result_array = np.array(result).reshape(-1, audio.channels).transpose().astype(np.float32)
        normalized_array = librosa.util.normalize(librosa.to_mono(result_array))

        mfcc_coefficients = librosa.feature.mfcc(
            y=normalized_array, 
            sr=audio.frame_rate, 
            n_mfcc=self._num_cepstral_coeffs)

        gfcc_coefficients = gfcc(
            sig=normalized_array, 
            fs=audio.frame_rate, 
            num_ceps=self._num_cepstral_coeffs)
        
        return (mfcc_coefficients, gfcc_coefficients.transpose())
           
    def __call__(self):
        for video_path in self._video_paths:
            for frame_grp in self._get_frames(video_path):
                yield frame_grp
 
# Configuration
FRAME_GROUP = 4 
IMG_SIZE = (112, 112) # Height, Width
VIDEO_PATHS = [base_path.joinpath("Family_Guy.mp4"), base_path.joinpath("Football_Clip.mp4")]

# audio.export(VIDEO_PATHS[0].cwd().joinpath("audio.wav"))
# y, sr = librosa.load(VIDEO_PATHS[0].cwd().joinpath("audio.wav"), sr=None)

# Instantiate the generator
frame_generator = FrameGenerator(
    video_paths=VIDEO_PATHS, 
    video_group_size=FRAME_GROUP, 
    target_shape=IMG_SIZE)

# Create the dataset
output_signature = tf.TensorSpec(shape=(FRAME_GROUP, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)

dataset = tf.data.Dataset.from_generator(
    frame_generator,
    output_signature=output_signature # or use output_signature
)

dataset = dataset.prefetch(tf.data.AUTOTUNE)

for video_grp in dataset:
    print("GENERATING DATA!")

##########################################################################################

input_tensor = layers.Input(shape=(FRAME_GROUP, 227, 227, 3)) # AlexNet typically uses 227x227x3 images

x = layers.Conv3D(filters=96, kernel_size=(1, 11, 11), strides=(1, 4, 4), activation='relu', padding='valid')(input_tensor)
x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='valid')(x)

x = layers.Conv3D(filters=256, kernel_size=(1, 5, 5), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='valid')(x)

x = layers.Conv3D(filters=384, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = layers.Conv3D(filters=384, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = layers.Conv3D(filters=256, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='valid')(x)

x = layers.Flatten()(x)
x = layers.Dense(units=4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(units=4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output_tensor = layers.Dense(units=NUM_CLASSES, activation='sigmoid')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)