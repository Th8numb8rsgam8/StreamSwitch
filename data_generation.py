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

class AdFrameGenerator:

    NUM_CEPSTRAL_FRAMES = 26 
    NUM_CEPSTRAL_COEFFS = 13

    def __init__(self, video_paths, video_group_size, target_shape=(224, 224)):

        self._video_paths = video_paths
        self._video_group_size = video_group_size
        self._target_shape = target_shape

        self._win_len = 0.025
        self._win_len_precision = 5

    def _get_frames(self, video_path):

        label_file = video_path.parent.joinpath(video_path.name.replace("mp4", "npy"))
        # label_array = np.load(label_file)

        video = cv2.VideoCapture(video_path)
        audio = AudioSegment.from_file(video_path, format="mp4", frame_rate=44100)

        if not video.isOpened():
            tf.print(f"Could not open {video_path}")
            return
            # sys.exit(1)

        total_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = video.get(cv2.CAP_PROP_FPS)
 
        num_groups, unused_video_frames = divmod(total_video_frames, self._video_group_size)
        audio_per_video, unused_audio_frames = divmod(audio.frame_count(), total_video_frames)
        audio_group_size = int(audio_per_video * self._video_group_size)

        # Select MFCC hop length
        mfcc_hop_length_lower = math.ceil(audio_group_size / AdFrameGenerator.NUM_CEPSTRAL_FRAMES)
        mfcc_hop_length_upper = math.floor(audio_group_size / (AdFrameGenerator.NUM_CEPSTRAL_FRAMES - 1))
        mfcc_hop_length_options = list(range(mfcc_hop_length_lower, mfcc_hop_length_upper + 1))
        mfcc_hop_length = random.choice(mfcc_hop_length_options)

        # Select GFCC window hop
        gfcc_win_hop_lower = (audio_group_size - audio.frame_rate * self._win_len) / (audio.frame_rate * AdFrameGenerator.NUM_CEPSTRAL_FRAMES)
        gfcc_win_hop_upper = (audio_group_size - audio.frame_rate * self._win_len) / (audio.frame_rate * (AdFrameGenerator.NUM_CEPSTRAL_FRAMES-1))
        gfcc_win_hop_lower = round(
            math.ceil(gfcc_win_hop_lower * 10**self._win_len_precision) / 10**self._win_len_precision, 
            self._win_len_precision)
        gfcc_win_hop_upper = round(
            math.floor(gfcc_win_hop_upper * 10**self._win_len_precision) / 10**self._win_len_precision, 
            self._win_len_precision)
        gfcc_win_hop = round(random.uniform(gfcc_win_hop_lower, gfcc_win_hop_upper), self._win_len_precision)

        tf.print(f"{video_path}: UNUSED VIDEO FRAMES {unused_video_frames}, UNUSED AUDIO FRAMES {unused_audio_frames}.")

        video_idx, audio_idx = 0, 0
        for _ in range(num_groups):
            video_frame_group = self._get_video_group(video)
            video_frame_group = tf.expand_dims(video_frame_group, axis=0)

            mfcc_coeffs, gfcc_coeffs = self._get_audio_group(
                audio, 
                audio_idx, 
                audio_idx + audio_group_size,
                mfcc_hop_length,
                gfcc_win_hop)
            audio_idx += audio_group_size

            audio_frame_group = tf.convert_to_tensor(np.concatenate((mfcc_coeffs, gfcc_coeffs), axis=0))
            audio_frame_group = tf.expand_dims(audio_frame_group, axis=2)
            audio_frame_group = tf.expand_dims(audio_frame_group, axis=0)

            labels = label_array[video_idx:video_idx+self._video_group_size]
            label = tf.expand_dims(np.argmax(np.bincount(labels)), axis=0)
            video_idx += self._video_group_size

            yield ((video_frame_group, audio_frame_group), label)
     
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
            n_mfcc=AdFrameGenerator.NUM_CEPSTRAL_COEFFS,
            hop_length=mfcc_hop_length)
        end_time = time.time()

        tf.print(f"MFCC TIME: {end_time-start_time}")

        window = SlidingWindow()
        window.win_hop = gfcc_win_hop

        start_time = time.time()
        gfcc_coefficients = gfcc(
            sig=normalized_array, 
            fs=audio.frame_rate, 
            num_ceps=AdFrameGenerator.NUM_CEPSTRAL_COEFFS, 
            window=window)
        end_time = time.time()

        tf.print(f"GFCC TIME: {end_time-start_time}")
        
        return (mfcc_coefficients, gfcc_coefficients.transpose())
           
    def __call__(self):
        for video_path in self._video_paths:
            for frame_grp in self._get_frames(video_path):
                yield frame_grp