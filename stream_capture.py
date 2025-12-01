import sys
import time
import pyaudio
import cv2
import threading
import numpy as np


class VideoCaptureThread(threading.Thread):

    def __init__(
        self, video_device_index=0, 
        frame_width=1280,
        frame_height=720):
        threading.Thread.__init__(self)

        self.cap = cv2.VideoCapture(video_device_index)
        if not self.cap.isOpened():
            print("Unabale to open capture device")
            sys.exit(1)

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.frame = None
        self.ret = False
        self.running = True
        self.frame_lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.frame_lock:
                self.ret = ret
                if ret:
                    self.frame = frame.copy()
            time.sleep(0.01)

    def read(self):
        with self.frame_lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


class AudioCaptureThread(threading.Thread):

    def __init__(
        self, 
        audio_device_index, 
        chunk=1024,
        audio_format=pyaudio.paInt16,
        channels=2,
        sample_rate=44100):

        self.stream = None

        threading.Thread.__init__(self)
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

        self.frames = []
        self.running = True

    def run(self):
        while self.running:
            data = self.stream.read(self.chunk)
            numpy_data = np.frombuffer(data, dtype=np.int16)
            self.frames.append(numpy_data)

    def get_data(self):

        frames_copy = self.frames.copy()
        self.frames.clear()
        return frames_copy

    def stop(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


def main():

    video_device_index = 0
    audio_device_index = 1

    video_thread = VideoCaptureThread(
        video_device_index,
        frame_width=512,
        frame_height=512)

    audio_thread = AudioCaptureThread(
        audio_device_index,
        chunk=2048)

    video_thread.start()
    audio_thread.start()

    try:
        while True:

            ret, frame = video_thread.read()

            if ret:
                cv2.imshow("HDMI Video Feed", frame)

            audio_data = audio_thread.get_data()
            if audio_data:
                print(f'Received {audio_data} audio chunks')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:

        video_thread.stop()
        audio_thread.stop()
        video_thread.join()
        audio_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
