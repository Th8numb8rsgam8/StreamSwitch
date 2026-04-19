import json
import math
import time
import random
import librosa
import scipy.io.wavfile as wavfile
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

import pdb

class AdFrameGenerator:

    NUM_CEPSTRAL_FRAMES = 52 
    NUM_CEPSTRAL_COEFFS = 26

    def __init__(self, video_paths, samples_per_video, sequence_length, target_shape=(224, 224)):

        self._video_paths = video_paths
        self._samples_per_video = samples_per_video
        self._sequence_length = sequence_length
        self._target_shape = target_shape

        self._win_len = 0.025
        self._win_len_precision = 5
        self._num_filters = 64

    def _get_frames(self, video_path):

        with open(video_path.joinpath("metadata.json"), "r") as f:
            metadata = json.load(f)
        video_frames = video_path.joinpath("video_frames")
        audio_file = video_path.joinpath("audio.wav")
        label_file = video_path.joinpath("labels.npy")
        label_tensor = tf.convert_to_tensor(np.load(label_file))
        audio_sample_rate, audio = wavfile.read(audio_file)

        total_video_frames = metadata["total_video_frames"]
        total_audio_frames = metadata["total_audio_frames"]
        audio_per_video, unused_audio_frames = divmod(total_audio_frames, total_video_frames)
        tf.print(f"{video_path}: UNUSED AUDIO FRAMES {unused_audio_frames}.")
        audio_per_video = int(audio_per_video)
        audio_frames_per_sequence = int(audio_per_video * self._sequence_length)
        max_video_idx = total_video_frames - self._sequence_length
        for sample_num in range(self._samples_per_video):
            start_video_idx = random.randint(0, max_video_idx)
            end_video_idx = start_video_idx + self._sequence_length

            video_frame_sequence = []
            for frame_idx in range(start_video_idx, end_video_idx):
                frame = tf.io.decode_jpeg(tf.io.read_file(str(video_frames.joinpath(f"frame_{frame_idx}.jpg"))), channels=3)
                frame = tf.reverse(frame, axis=[-1]) # RGB to BGR
                frame = tf.image.convert_image_dtype(frame, tf.float32)
                frame = tf.image.resize(frame, self._target_shape)
                video_frame_sequence.append(frame)

            video_sequence = tf.stack(video_frame_sequence, axis=0)

            # Select MFCC hop length
            mfcc_hop_length = self._get_mfcc_hop_length(audio_frames_per_sequence)

            # Select GFCC window hop
            gfcc_win_hop = self._get_gfcc_window_hop(audio_frames_per_sequence, audio_sample_rate)
        
            start_audio_idx = start_video_idx * audio_per_video
            mfcc_coeffs, gfcc_coeffs = self._get_audio_group(
                audio, 
                audio_sample_rate,
                start_audio_idx, 
                start_audio_idx + audio_frames_per_sequence,
                mfcc_hop_length,
                gfcc_win_hop)

            try:
                audio_spectrogram = tf.convert_to_tensor(np.concatenate((mfcc_coeffs, gfcc_coeffs), axis=0), dtype=tf.float32)
            except ValueError as e:
                tf.print(f"DIMENSION MISMATCH - MFCC {mfcc_coeffs.shape[1]}, GFCC {gfcc_coeffs.shape[1]}")
                if gfcc_coeffs.shape[1] > mfcc_coeffs.shape[1]:
                    audio_spectrogram = tf.convert_to_tensor(np.concatenate((mfcc_coeffs, gfcc_coeffs[:,:-1]), axis=0), dtype=tf.float32)
                else:
                    audio_spectrogram = tf.convert_to_tensor(np.concatenate((mfcc_coeffs[:,:-1], gfcc_coeffs), axis=0), dtype=tf.float32)

            audio_spectrogram = tf.expand_dims(audio_spectrogram, axis=2)

            label = label_tensor[end_video_idx]
            
            yield ((video_sequence, audio_spectrogram), label)
    
    def _get_video_group(self, video):

        frame_grp = []
        for _ in range(self._video_group_size):
            ret, frame = video.read()
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            frame = tf.image.resize(frame, self._target_shape)
            frame_grp.append(frame)
        return tf.stack(frame_grp, axis=0)

    def _get_mfcc_hop_length(self, audio_frames_per_sequence):

        mfcc_hop_length_lower = math.ceil(audio_frames_per_sequence / AdFrameGenerator.NUM_CEPSTRAL_FRAMES)
        mfcc_hop_length_upper = math.floor(audio_frames_per_sequence / (AdFrameGenerator.NUM_CEPSTRAL_FRAMES - 1))
        mfcc_hop_length_options = list(range(mfcc_hop_length_lower, mfcc_hop_length_upper + 1))
        mfcc_hop_length = random.choice(mfcc_hop_length_options)

        return mfcc_hop_length
    
    def _get_gfcc_window_hop(self, audio_frames_per_sequence, audio_sample_rate):

        gfcc_win_hop_lower = (audio_frames_per_sequence - audio_sample_rate * self._win_len) / (audio_sample_rate * AdFrameGenerator.NUM_CEPSTRAL_FRAMES)
        gfcc_win_hop_upper = (audio_frames_per_sequence - audio_sample_rate * self._win_len) / (audio_sample_rate * (AdFrameGenerator.NUM_CEPSTRAL_FRAMES-1))
        gfcc_win_hop_lower = round(math.ceil(gfcc_win_hop_lower * 10**self._win_len_precision) / 10**self._win_len_precision, self._win_len_precision)
        gfcc_win_hop_upper = round(math.floor(gfcc_win_hop_upper * 10**self._win_len_precision) / 10**self._win_len_precision, self._win_len_precision)
        gfcc_win_hop = round(random.uniform(gfcc_win_hop_lower, gfcc_win_hop_upper), self._win_len_precision)

        return gfcc_win_hop
    
    def _get_audio_group(self, audio, audio_sample_rate, start_idx, end_idx, mfcc_hop_length, gfcc_win_hop):

        audio_array = audio[start_idx:end_idx, :].transpose().astype(np.float32)
        normalized_array = librosa.util.normalize(librosa.to_mono(audio_array))

        mfcc_coefficients = librosa.feature.mfcc(
            y=normalized_array, 
            sr=audio_sample_rate, 
            n_mfcc=AdFrameGenerator.NUM_CEPSTRAL_COEFFS,
            hop_length=mfcc_hop_length)

        window = SlidingWindow()
        window.win_hop = gfcc_win_hop

        gfcc_coefficients = gfcc(
            sig=normalized_array, 
            fs=audio_sample_rate, 
            num_ceps=AdFrameGenerator.NUM_CEPSTRAL_COEFFS,
            nfilts=self._num_filters,
            window=window)
        
        return (mfcc_coefficients, gfcc_coefficients.transpose())
           
    def __call__(self):
        for video_path in self._video_paths:
            for frame_grp in self._get_frames(video_path):
                yield frame_grp