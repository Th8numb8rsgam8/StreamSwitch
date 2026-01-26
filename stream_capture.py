import sys
import time
from datetime import datetime
import pyaudio
import librosa
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import tensorflow as tf

import pdb

class VideoCaptureProcess(mp.Process):

    def __init__(
        self, video_device_index=0, 
        frame_width=1280,
        frame_height=720,
        synchronizer=None,
        deposit_name=None,
        *args, **kwargs):

        super(VideoCaptureProcess, self).__init__(*args, **kwargs)

        self.synchronizer = synchronizer
        self.frame_deposit_buffer = shared_memory.SharedMemory(name=deposit_name)
        self.frame_deposit_array = np.ndarray(
            frame_height * frame_width * 3, 
            dtype=np.uint8, 
            buffer=self.frame_deposit_buffer.buf).reshape(frame_height, frame_width, 3)

        self.cap = cv2.VideoCapture(video_device_index)
        if not self.cap.isOpened():
            print("Unabale to open capture device")
            sys.exit(1)

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    def run(self):
        self.synchronizer.wait()
        process_name = mp.current_process().name
        while True:
            # time_prev = datetime.now()
            ret, self.frame_deposit_array[:] = self.cap.read()
            # time_next = datetime.now()
            # print(f"{process_name}: {time_next-time_prev}")

    # def stop(self):
    #     self.running = False
    #     self.cap.release()


class AudioCaptureProcess(mp.Process):

    def __init__(
        self, 
        audio_device_index, 
        chunk=1024,
        audio_format=pyaudio.paInt16,
        channels=2,
        sample_rate=44100,
        synchronizer=None,
        deposit_name=None,
        *args, **kwargs):

        super(AudioCaptureProcess, self).__init__(*args, **kwargs)

        self.synchronizer = synchronizer
        self.frame_deposit_buffer = shared_memory.SharedMemory(name=deposit_name)
        self.frame_deposit_array = np.ndarray(
            chunk * channels, 
            dtype=np.int16, 
            buffer=self.frame_deposit_buffer.buf).reshape(channels, -1)

        self.stream = None
        self.p = pyaudio.PyAudio()
        audio_info = self.p.get_default_input_device_info()

        self.chunk = chunk
        self.audio_format = audio_format
        self.channels = channels
        self.sample_rate = sample_rate

        try:
            self.stream = self.p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=int(audio_info["defaultSampleRate"]),
                input=True,
                input_device_index=audio_device_index,
                frames_per_buffer=self.chunk)

        except OSError as e:
            if e.strerror == "Invalid sample rate":
                self.stream = self.p.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=48000,
                    input=True,
                    input_device_index=audio_device_index,
                    frames_per_buffer=self.chunk)

    def run(self):
        self.synchronizer.wait()
        process_name = mp.current_process().name
        while True:
            # time_prev = datetime.now()
            data = self.stream.read(self.chunk)
            numpy_data = np.frombuffer(data, dtype=np.int16)
            audio_frames = numpy_data.reshape(-1, self.channels).transpose()
            self.frame_deposit_array[:] = audio_frames
            # time_next = datetime.now()
            # print(f"{process_name}: {time_next-time_prev}")

    # def stop(self):
    #     self.running = False
    #     self.stream.stop_stream()
    #     self.stream.close()
    #     self.p.terminate()


def main():

    try:
        VIDEO_DEVICE_INDEX = 0
        AUDIO_DEVICE_INDEX = 1
        stream_synchronizer = mp.Event()

        VIDEO_FRAME_SHAPE = (480, 640)
        AUDIO_CHUNK_SIZE = 2048
        NUM_AUDIO_CHANNELS = 2
        NUM_CEPSTRAL_FRAMES = 26 
        NUM_CEPSTRAL_COEFFS = 13

        video_frame_template = np.ndarray(
            VIDEO_FRAME_SHAPE[0] * VIDEO_FRAME_SHAPE[1] * 3,
            dtype=np.uint8)

        audio_frame_template = np.ndarray(
            AUDIO_CHUNK_SIZE * NUM_AUDIO_CHANNELS, dtype=np.int16)

        video_shared_memory = shared_memory.SharedMemory(
            create=True, 
            size=video_frame_template.nbytes)

        audio_shared_memory = shared_memory.SharedMemory(
            create=True, 
            size=audio_frame_template.nbytes)

        video_frame_retrieval = np.ndarray(
            VIDEO_FRAME_SHAPE[0] * VIDEO_FRAME_SHAPE[1] * 3, 
            dtype=np.uint8, 
            buffer=video_shared_memory.buf).reshape(*VIDEO_FRAME_SHAPE, 3)

        audio_frame_retrieval= np.ndarray(
            AUDIO_CHUNK_SIZE * NUM_AUDIO_CHANNELS, 
            dtype=np.int16, 
            buffer=audio_shared_memory.buf).reshape(NUM_AUDIO_CHANNELS, -1)

        video_process = VideoCaptureProcess(
            name="Video Capture Process",
            video_device_index=VIDEO_DEVICE_INDEX, 
            frame_width=VIDEO_FRAME_SHAPE[1], 
            frame_height=VIDEO_FRAME_SHAPE[0],
            synchronizer=stream_synchronizer,
            deposit_name=video_shared_memory.name)

        audio_process = AudioCaptureProcess(
            name="Audio Capture Process",
            audio_device_index=AUDIO_DEVICE_INDEX, 
            chunk=AUDIO_CHUNK_SIZE,
            channels=NUM_AUDIO_CHANNELS,
            synchronizer=stream_synchronizer,
            deposit_name=audio_shared_memory.name)

        video_process.start()
        audio_process.start()

        stream_synchronizer.set()

        frame_pair = []
        window = SlidingWindow()
        while True:
            if len(frame_pair) == 2:
                print(np.array_equal(frame_pair[0], frame_pair[1]))
                frame_pair.clear()
            current_time = datetime.now().strftime('%H:%M:%S.%f')
            process_name = mp.current_process().name
            # frame_pair.append(video_frame_retrieval.copy())
            # time.sleep(0.01)
            frame = tf.image.convert_image_dtype(video_frame_retrieval, tf.float32)
            frame = tf.image.resize(frame, (224, 224))

            print(f"FIDEO FRAME {frame.shape}")

            normalized_array = librosa.util.normalize(
                librosa.to_mono(audio_frame_retrieval.astype(np.float32)))

            mfcc_coefficients = librosa.feature.mfcc(
                y=normalized_array, 
                sr=48000,
                n_mfcc=NUM_CEPSTRAL_COEFFS,
                hop_length=512)

            # window.win_hop = gfcc_win_hop

            gfcc_coefficients = gfcc(
                sig=normalized_array, 
                fs=48000, 
                num_ceps=NUM_CEPSTRAL_COEFFS, 
                window=window).transpose()

            print(f"MFCC SHAPE: {mfcc_coefficients.shape}")
            print(f"GFCC SHAPE: {gfcc_coefficients.shape}")
            # print(f"{process_name} {current_time}: VIDEO FRAME {video_frame_retrieval.shape}")
            # print(f"{process_name} {current_time}: AUDIO FRAME {audio_frame_retrieval.shape}")

    except KeyboardInterrupt as e:
        print("PROGRAM ABORTED!")


if __name__ == "__main__":
    main()
