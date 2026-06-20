import tensorflow as tf
import numpy as np

import pdb

class ParallelFrameGenerator:

    def __init__(
        self, 
        label_array_table,
        ragged_labels_data,
        video_frame_count_table, 
        audio_frame_count_table, 
        sequence_length, 
        video_frames_per_audio,
        target_shape=(224, 224)):

        self._label_array_table = label_array_table
        self._ragged_labels_data = ragged_labels_data
        self._video_frame_count_table = video_frame_count_table
        self._audio_frame_count_table = audio_frame_count_table
        self._sequence_length = sequence_length
        self._target_shape = target_shape

        self._VIDEO_FRAME_RATE = tf.cast(30 ,dtype=tf.float32)
        self._AUDIO_SAMPLE_RATE = tf.cast(16000, dtype=tf.float32)
        self._NUM_AUDIO_CHANNELS = tf.cast(2, dtype=tf.int32)
        # self._NUM_CEPSTRAL_FRAMES = tf.cast(208 , dtype=tf.int32)
        # self._NUM_CEPSTRAL_COEFFS = tf.cast(26, dtype=tf.int32)
        # self._WINDOW_LENGTH = 0.128
        # self._N_TAPS = 2048
        # self._NUM_FILTERS = 64
        # self._FMIN, self._FMAX = 50, self._AUDIO_SAMPLE_RATE / 2
        # self._GTCC_FILTERBANK = self._create_gtcc_filterbank()
        # self._MFCC_FILTERBANK = self._create_mfcc_filterbank()
        # self._win_len_precision = 5

        self._IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self._IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

        self._YAMNET_FRAME_TIME = tf.cast(0.96, dtype=tf.float32) # seconds
        self._YAMNET_HOP_TIME = 0.48 # seconds
        self._YAMNET_SEQUENCE_TIME = self._YAMNET_FRAME_TIME + (sequence_length - 1) * self._YAMNET_HOP_TIME
        self._YAMNET_FRAME_SAMPLES = tf.cast(self._YAMNET_FRAME_TIME * self._AUDIO_SAMPLE_RATE, dtype=tf.int32)
        self._YAMNET_SEQUENCE_SAMPLES = tf.cast(self._AUDIO_SAMPLE_RATE * self._YAMNET_SEQUENCE_TIME, dtype=tf.int32)

        self._VIDEO_FRAMES_PER_AUDIO = tf.cast(video_frames_per_audio, dtype=tf.int32)
        self._VIDEO_FRAME_TIME = tf.cast(self._YAMNET_FRAME_TIME / tf.cast(self._VIDEO_FRAMES_PER_AUDIO, dtype=tf.float32), dtype=tf.float32)
        self._AUDIO_SAMPLES_OVERFLOW = tf.cast(tf.cast(0.5, dtype=tf.float32) * self._VIDEO_FRAME_TIME * self._AUDIO_SAMPLE_RATE, dtype=tf.int32)
        self._VIDEO_FRAME_HOP = tf.cast(tf.math.floor((self._YAMNET_FRAME_TIME * self._VIDEO_FRAME_RATE) / tf.cast(self._VIDEO_FRAMES_PER_AUDIO, dtype=tf.float32)), dtype=tf.int32)

    def get_frame(self, video_path):

        video_frames = tf.strings.join([video_path, "video_frames"], separator="/")
        audio_file = tf.strings.join([video_path, "audio_16khz.wav"], separator="/")
        label_file = tf.strings.join([video_path, "labels.npy"], separator="/")

        audio_binary = tf.io.read_file(audio_file)
        audio, audio_sample_rate = tf.audio.decode_wav(audio_binary)

        total_video_frames = self._video_frame_count_table.lookup(video_path)
        total_audio_frames = self._audio_frame_count_table.lookup(video_path)
        extra_audio_frames = tf.math.floormod(total_audio_frames, total_video_frames)

        padding_size = total_video_frames - extra_audio_frames
        padding = tf.zeros((padding_size, self._NUM_AUDIO_CHANNELS), dtype=tf.float32)
        audio = tf.concat([audio, padding], axis=0)
        audio_per_video = tf.math.floordiv(total_audio_frames + padding_size, total_video_frames)
        audio_per_video = tf.cast(audio_per_video, dtype=tf.int32)
        # audio_frames_per_sequence = tf.cast(audio_per_video * self._sequence_length, dtype=tf.int32)
        # max_video_idx = total_video_frames - self._sequence_length
        min_video_idx = tf.cast(tf.math.ceil(self._AUDIO_SAMPLES_OVERFLOW / audio_per_video) + 1, dtype=tf.int32)
        max_video_idx = total_video_frames - self._sequence_length * self._VIDEO_FRAMES_PER_AUDIO * self._VIDEO_FRAME_HOP

        start_video_idx = tf.random.uniform(shape=[], minval=min_video_idx, maxval=max_video_idx+1, dtype=tf.int32)
        end_video_idx = start_video_idx + self._sequence_length * self._VIDEO_FRAMES_PER_AUDIO * self._VIDEO_FRAME_HOP
        sequence_indices = tf.range(start_video_idx, end_video_idx, self._VIDEO_FRAME_HOP)

        image_paths = tf.vectorized_map(lambda x: tf.strings.join([video_frames, "frame_" + tf.strings.as_string(x) + ".jpg"], separator="/"), sequence_indices)
        video_sequence = tf.vectorized_map(self._get_video_sequence, image_paths)
        # video_sequence = self._normalize_data(video_sequence)

        # Select window hop
        # frame_step = self._get_frame_step(audio_frames_per_sequence)
        
        start_audio_idx = start_video_idx * audio_per_video - self._AUDIO_SAMPLES_OVERFLOW
        end_audio_idx = start_audio_idx + self._YAMNET_SEQUENCE_SAMPLES
        indices = tf.range(start_audio_idx, end_audio_idx)
        audio_segment = tf.gather(audio, indices, axis=0)
        audio_segment = tf.transpose(audio_segment)
        audio_segment = tf.reduce_mean(audio_segment, axis=0)
        # processed_audio_segment = self._get_processed_audio(audio_segment, frame_step)

        index = self._label_array_table.lookup(video_path)
        label_array = tf.gather(self._ragged_labels_data, index)
        labels = tf.gather(label_array, sequence_indices, axis=0)

        transition_check = tf.equal(labels, 2)
        contains_transition = tf.reduce_any(transition_check)
        if contains_transition:
            label = 2
        else:
            unique_values, _, counts = tf.compat.v1.unique_with_counts(labels)
            max_idx = tf.math.argmax(counts)
            label = unique_values[max_idx]

        return (audio_segment, video_sequence), label
    
    def _get_video_sequence(self, video_path):

        frame = tf.io.decode_jpeg(tf.io.read_file(video_path), channels=3)
        # frame = tf.reverse(frame, axis=[-1]) # RGB to BGR
        # frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = tf.image.resize(frame, self._target_shape)
        # frame = (frame - self._IMAGENET_MEAN) / self._IMAGENET_STD

        return frame

    def _create_mfcc_filterbank(self):

        mfcc_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self._NUM_FILTERS, 
            num_spectrogram_bins=(self._N_TAPS // 2 + 1), 
            sample_rate=self._AUDIO_SAMPLE_RATE, 
            lower_edge_hertz=self._FMIN, 
            upper_edge_hertz=self._FMAX
        )

        return mfcc_filterbank

    def _create_gtcc_filterbank(self):

        erb_min = 21.4 * np.log10(4.37e-3 * self._FMIN + 1)
        erb_max = 21.4 * np.log10(4.37e-3 * self._FMAX + 1)
        erb_pts = np.linspace(erb_min, erb_max, self._NUM_FILTERS)
        f_centers = (10**(erb_pts / 21.4) - 1) / 4.37e-3

        gtcc_filterbank = np.zeros((len(f_centers), self._N_TAPS // 2 + 1))
        for i, fc in enumerate(f_centers):
            fc = fc.astype(int) - .0001
            fir_magnitude = self._design_gammatone_fir(fc)
            gtcc_filterbank[i,:] = fir_magnitude
        
        return tf.transpose(tf.convert_to_tensor(gtcc_filterbank, dtype=tf.float32))

    def _design_gammatone_fir(self, fc, order=4):

        t = np.arange(self._N_TAPS) / self._AUDIO_SAMPLE_RATE
        erb = 24.7 * (4.37 * fc / 1000 + 1)
        b = 1.019 * erb

        gt_ir = (t**(order-1)) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t)
        gt_ir = gt_ir / np.sum(np.abs(gt_ir))
        fir_coeffs = np.fft.rfft(gt_ir)

        magnitude = np.abs(fir_coeffs)

        return magnitude
    
    def _get_frame_step(self, audio_frames_per_sequence):

        audio_len = tf.cast(audio_frames_per_sequence, dtype=tf.float32)
        num_cepstral_frames = tf.cast(self._NUM_CEPSTRAL_FRAMES, dtype=tf.float32)
        win_hop_lower = (audio_len - self._AUDIO_SAMPLE_RATE * self._WINDOW_LENGTH) / (self._AUDIO_SAMPLE_RATE * num_cepstral_frames)
        win_hop_upper = (audio_len - self._AUDIO_SAMPLE_RATE * self._WINDOW_LENGTH) / (self._AUDIO_SAMPLE_RATE * (num_cepstral_frames - 1.0))
        win_hop_lower = tf.math.ceil(win_hop_lower * 10**5) / 10**5
        win_hop_upper = tf.math.floor(win_hop_upper * 10**5) / 10**5
        win_hop = tf.random.uniform(
            shape=[],
            minval=win_hop_lower, 
            maxval=win_hop_upper,
            dtype=tf.float32)

        frame_step = tf.cast(win_hop * self._AUDIO_SAMPLE_RATE, tf.int32)
        return frame_step

    def _normalize_data(self, data):

        mean = tf.math.reduce_mean(data)
        variance = tf.math.reduce_variance(data)

        epsilon = 1e-7  # Prevents division by zero
        std_dev = tf.math.sqrt(variance + epsilon)
        normalized_data = (data - mean) / std_dev

        return normalized_data

    def _get_processed_audio(self, audio_segment, frame_step):

        audio_segment = tf.transpose(audio_segment)
        audio_segment = tf.cast(audio_segment[...], dtype=tf.float32)
        audio_segment = tf.reduce_mean(audio_segment, axis=0)
        audio_segment, _ = tf.linalg.normalize(audio_segment)
        emphasis = audio_segment[..., 1:] - 0.97 * audio_segment[..., :-1]
        audio_segment = tf.concat([audio_segment[...,:1], emphasis], axis=0)

        stft = tf.signal.stft(
            audio_segment,
            frame_length=self._N_TAPS,
            frame_step=frame_step,
            window_fn=tf.signal.hamming_window
        )

        spectrogram = tf.abs(stft)

        mel_spectrogram = tf.tensordot(spectrogram, self._MFCC_FILTERBANK, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfcc_coefficients = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfcc_coefficients = mfcc_coefficients[..., :self._NUM_CEPSTRAL_FRAMES, :self._NUM_CEPSTRAL_COEFFS]
        mfcc_coefficients = self._normalize_data(mfcc_coefficients)

        gammatone_spectrogram = tf.tensordot(spectrogram, self._GTCC_FILTERBANK, 1)
        log_gammatone_spectrogram = tf.math.log(gammatone_spectrogram + 1e-10)
        gtcc_coefficients = tf.signal.dct(log_gammatone_spectrogram, type=2, norm='ortho')
        gtcc_coefficients = gtcc_coefficients[..., :self._NUM_CEPSTRAL_FRAMES, :self._NUM_CEPSTRAL_COEFFS]
        gtcc_coefficients = self._normalize_data(gtcc_coefficients)

        processed_audio_segment = tf.stack([mfcc_coefficients, gtcc_coefficients], axis=-1)

        return processed_audio_segment