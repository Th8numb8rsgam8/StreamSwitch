import os
import math
import time
import random
import cv2
import librosa
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
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

    NUM_CEPSTRAL_FRAMES = 20 
    NUM_CEPSTRAL_COEFFS = 13

    def __init__(self, video_paths, video_group_size, target_shape=(224, 224)):

        self._video_paths = video_paths
        self._video_group_size = video_group_size
        self._target_shape = target_shape

        self._win_len = 0.025
        self._win_len_precision = 5

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
        audio_per_video, unused_audio_frames = divmod(audio.frame_count(), total_video_frames)
        audio_group_size = audio_per_video * self._video_group_size

        # Select MFCC hop length
        mfcc_hop_length_lower = math.ceil(audio_group_size / FrameGenerator.NUM_CEPSTRAL_FRAMES)
        mfcc_hop_length_upper = math.floor(audio_group_size / (FrameGenerator.NUM_CEPSTRAL_FRAMES - 1))
        mfcc_hop_length_options = list(range(mfcc_hop_length_lower, mfcc_hop_length_upper + 1))
        mfcc_hop_length = random.choice(mfcc_hop_length_options)

        # Select GFCC window hop
        gfcc_win_hop_lower = (audio_group_size - audio.frame_rate * self._win_len) / (audio.frame_rate * FrameGenerator.NUM_CEPSTRAL_FRAMES)
        gfcc_win_hop_upper = (audio_group_size - audio.frame_rate * self._win_len) / (audio.frame_rate * (FrameGenerator.NUM_CEPSTRAL_FRAMES-1))
        gfcc_win_hop_lower = round(
            math.ceil(gfcc_win_hop_lower * 10**self._win_len_precision) / 10**self._win_len_precision, 
            self._win_len_precision)
        gfcc_win_hop_upper = round(
            math.floor(gfcc_win_hop_upper * 10**self._win_len_precision) / 10**self._win_len_precision, 
            self._win_len_precision)
        gfcc_win_hop = round(random.uniform(gfcc_win_hop_lower, gfcc_win_hop_upper), self._win_len_precision)

        print(f"{video_path}: UNUSED VIDEO FRAMES {unused_video_frames}, UNUSED AUDIO FRAMES {unused_audio_frames}.")

        audio_idx = 0
        for _ in range(num_groups):
            video_frame_group = self._get_video_group(video)
            mfcc_coeffs, gfcc_coeffs = self._get_audio_group(
                audio, 
                audio_idx, 
                audio_idx + audio_group_size,
                mfcc_hop_length,
                gfcc_win_hop)
            audio_idx += audio_group_size

            audio_frame_group = tf.convert_to_tensor(np.concatenate((mfcc_coeffs, gfcc_coeffs), axis=0))

            yield (video_frame_group, audio_frame_group)
     
        video.release()
    
    def _get_video_group(self, video):

        frame_grp = []
        for _ in range(self._video_group_size):
            ret, frame = video.read()
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            frame = tf.image.resize(frame, self._target_shape)
            frame_grp.append(frame)
        return tf.stack(frame_grp, axis=0)
    
    def _get_audio_group(self, audio, start_idx, end_idx, mfcc_hop_length, gfcc_win_hop):

        result = audio.get_sample_slice(start_idx, end_idx).get_array_of_samples()
        result_array = np.array(result).reshape(-1, audio.channels).transpose().astype(np.float32)
        normalized_array = librosa.util.normalize(librosa.to_mono(result_array))

        start_time = time.time()
        mfcc_coefficients = librosa.feature.mfcc(
            y=normalized_array, 
            sr=audio.frame_rate, 
            n_mfcc=FrameGenerator.NUM_CEPSTRAL_COEFFS,
            hop_length=mfcc_hop_length)
        end_time = time.time()

        print(f"MFCC TIME: {end_time-start_time}")

        window = SlidingWindow()
        window.win_hop = gfcc_win_hop

        start_time = time.time()
        gfcc_coefficients = gfcc(
            sig=normalized_array, 
            fs=audio.frame_rate, 
            num_ceps=FrameGenerator.NUM_CEPSTRAL_COEFFS, 
            window=window)
        end_time = time.time()

        print(f"GFCC TIME: {end_time-start_time}")
        
        return (mfcc_coefficients, gfcc_coefficients.transpose())
           
    def __call__(self):
        for video_path in self._video_paths:
            for frame_grp in self._get_frames(video_path):
                yield frame_grp
 
# Configuration
FRAME_GROUP = 4 
IMG_SIZE = (112, 112) # Height, Width
VIDEO_PATHS = [base_path.joinpath("Family_Guy.mp4"), base_path.joinpath("Football_Clip.mp4")]

# Instantiate the generator
frame_generator = FrameGenerator(
    video_paths=VIDEO_PATHS, 
    video_group_size=FRAME_GROUP, 
    target_shape=IMG_SIZE)

# Create the dataset
output_signature = (
    tf.TensorSpec(shape=(FRAME_GROUP, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
    tf.TensorSpec(shape=(2 * FrameGenerator.NUM_CEPSTRAL_COEFFS, FrameGenerator.NUM_CEPSTRAL_FRAMES), dtype=tf.float32)
)

dataset = tf.data.Dataset.from_generator(
    frame_generator,
    output_signature=output_signature # or use output_signature
)

dataset = dataset.prefetch(tf.data.AUTOTUNE)

# for video_grp, audio_grp in dataset:
#     pdb.set_trace()
#     print("GENERATING DATA!")

# VIDEO NEURAL NET
##########################################################################################

# video_input_tensor = layers.Input(
#     shape=(FRAME_GROUP, 227, 227, 3), 
#     name='video_input')
# 
# x = layers.Conv3D(
#     filters=96, 
#     kernel_size=(1, 11, 11), 
#     strides=(1, 4, 4), 
#     activation='relu', 
#     padding='valid',
#     name="video_conv_1")(video_input_tensor)
# 
# x = layers.MaxPooling3D(
#     pool_size=(1, 3, 3), 
#     strides=(1, 2, 2), 
#     padding='valid',
#     name="video_pool_1")(x)
# 
# x = layers.Conv3D(
#     filters=256, 
#     kernel_size=(1, 5, 5), 
#     strides=(1, 1, 1), 
#     activation='relu', 
#     padding='same',
#     name="video_conv_2")(x)
# 
# x = layers.MaxPooling3D(
#     pool_size=(1, 3, 3), 
#     strides=(1, 2, 2), 
#     padding='valid',
#     name="video_pool_2")(x)
# 
# x = layers.Conv3D(filters=384, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name="video_conv_3")(x)
# x = layers.Conv3D(filters=384, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name="video_conv_4")(x)
# x = layers.Conv3D(filters=256, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name="video_conv_5")(x)
# x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='valid', name="video_pool_3")(x)
# 
# x = layers.Flatten()(x)
# x = layers.Dense(units=4096, activation='relu', name="video_dense_1")(x)
# x = layers.Dropout(0.5, name="video_dropout_1")(x)
# x = layers.Dense(units=4096, activation='relu', name="video_dense_2")(x)
# x = layers.Dropout(0.5, name="video_dropout_2")(x)
# 
# output_tensor = layers.Dense(units=NUM_CLASSES, activation='sigmoid')(x)
# model = Model(inputs=video_input_tensor, outputs=output_tensor)

# AUDIO NEURAL NET
##########################################################################################

audio_input_tensor = layers.Input(
    shape=(FrameGenerator.NUM_CEPSTRAL_COEFFS, FrameGenerator.NUM_CEPSTRAL_FRAMES), 
    name='audio_input')

x = layers.Conv2D(
    filters=96, 
    kernel_size=(11, 11), 
    strides=(4, 4), 
    activation='relu', 
    padding='valid',
    name="audio_conv_1")(audio_input_tensor)

x = layers.MaxPooling2D(
    pool_size=(3, 3), 
    strides=(2, 2), 
    padding='valid',
    name="audio_pool_1")(x)

x = layers.Conv2D(
    filters=256, 
    kernel_size=(5, 5), 
    strides=(1, 1), 
    activation='relu', 
    padding='same',
    name="audio_conv_2")(x)

x = layers.MaxPooling2D(
    pool_size=(3, 3), 
    strides=(2, 2), 
    padding='valid',
    name="audio_pool_2")(x)

x = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name="audio_conv_3")(x)
x = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name="audio_conv_4")(x)
x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name="audio_conv_5")(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name="audio_pool_3")(x)

x = layers.Flatten()(x)
x = layers.Dense(units=4096, activation='relu', name="audio_dense_1")(x)
x = layers.Dropout(0.5, name="audio_dropout_1")(x)
x = layers.Dense(units=4096, activation='relu', name="audio_dense_2")(x)
x = layers.Dropout(0.5, name="audio_dropout_2")(x)

output_tensor = layers.Dense(units=NUM_CLASSES, activation='sigmoid')(x)
model = Model(inputs=audio_input_tensor, outputs=output_tensor)

pdb.set_trace()