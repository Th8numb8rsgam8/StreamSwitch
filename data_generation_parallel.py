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
        target_shape=(224, 224)):

        self._label_array_table = label_array_table
        self._ragged_labels_data = ragged_labels_data
        self._video_frame_count_table = video_frame_count_table
        self._audio_frame_count_table = audio_frame_count_table
        self._sequence_length = sequence_length
        self._target_shape = target_shape

        self._AUDIO_SAMPLE_RATE = tf.cast(16000, dtype=tf.float32)
        self._NUM_AUDIO_CHANNELS = tf.cast(2, dtype=tf.int32)
        self._NUM_CEPSTRAL_FRAMES = tf.cast(104 , dtype=tf.int32)
        self._NUM_CEPSTRAL_COEFFS = tf.cast(26, dtype=tf.int32)
        self._WINDOW_LENGTH = 0.128
        self._N_TAPS = 2048
        self._NUM_FILTERS = 64
        self._FMIN, self._FMAX = 50, self._AUDIO_SAMPLE_RATE / 2
        self._GTCC_FILTERBANK = self._create_gtcc_filterbank()
        self._MFCC_FILTERBANK = self._create_mfcc_filterbank()

        self._win_len_precision = 5

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
        audio_frames_per_sequence = tf.cast(audio_per_video * self._sequence_length, dtype=tf.int32)
        max_video_idx = total_video_frames - self._sequence_length

        start_video_idx = tf.random.uniform(shape=[], minval=0, maxval=max_video_idx+1, dtype=tf.int32)
        end_video_idx = start_video_idx + self._sequence_length
        sequence_indices = tf.range(start_video_idx, end_video_idx)

        image_paths = tf.vectorized_map(lambda x: tf.strings.join([video_frames, "frame_" + tf.strings.as_string(x) + ".jpg"], separator="/"), sequence_indices)
        video_sequence = tf.vectorized_map(self._get_video_sequence, image_paths)
        video_sequence = self._normalize_data(video_sequence)

        # Select window hop
        frame_step = self._get_frame_step(audio_frames_per_sequence)
        
        start_audio_idx = start_video_idx * audio_per_video
        end_audio_idx = start_audio_idx + audio_frames_per_sequence
        indices = tf.range(start_audio_idx, end_audio_idx)
        audio_segment = tf.gather(audio, indices, axis=0)
        processed_audio_segment = self._get_processed_audio(audio_segment, frame_step)

        index = self._label_array_table.lookup(video_path)
        label_array = tf.gather(self._ragged_labels_data, index)
        label = label_array[end_video_idx-1]

        return (video_sequence, processed_audio_segment), label
    
    def _get_video_sequence(self, video_path):

        frame = tf.io.decode_jpeg(tf.io.read_file(video_path), channels=3)
        frame = tf.reverse(frame, axis=[-1]) # RGB to BGR
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        frame = tf.image.resize(frame, self._target_shape)

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

################################################################################

# import numpy as np
# 
# def gammatone_iir_coefficients(fc, fs):
#     """Calculates coefficients for a 4th-order IIR gammatone filter."""
#     erb = 24.7 * (4.37e-3 * fc + 1.0)  # Equivalent Rectangular Bandwidth
#     t_pt = 1.0 / fs
#     b = 1.019 * 2 * np.pi * erb
#     
#     # Complex pole for the 1st-order section
#     phi = 2 * np.pi * fc * t_pt
#     rho = np.exp(-b * t_pt)
#     
#     # 4th-order IIR is a cascade of four identical 1st-order complex filters
#     # Transfer function: H(z) = [ (1 - rho*e^{j*phi}*z^-1) ]^-4
#     # The coefficients below represent the denominator of the real-valued filter
#     a = np.array([1, -4*rho*np.cos(phi), 
#                   rho**2 * (2 + 4*np.cos(phi)**2), 
#                   -4*rho**3*np.cos(phi), rho**4])
#     b_coeff = np.array([1.0]) # Simplified gain
#     return b_coeff, a
# 
# def apply_iir_filter(b, a, x):
#     """Manual IIR filter implementation (Direct Form II)."""
#     y = np.zeros_like(x)
#     w = np.zeros(len(a))
#     for n in range(len(x)):
#         # x[n] - sum(a[1:] * w[1:])
#         w[0] = x[n] - np.dot(a[1:], w[1:])
#         y[n] = np.dot(b, w[:len(b)])
#         # Shift delay line
#         w[1:] = w[:-1]
#     return y
# 
# def get_gtcc(signal, fs, num_ceps=13, num_filters=26):
#     """Extracts Gammatone Cepstral Coefficients (GTCC)."""
#     # 1. Design filterbank (linear on ERB scale)
#     f_min, f_max = 50, fs / 2
#     erb_min = 21.4 * np.log10(4.37e-3 * f_min + 1)
#     erb_max = 21.4 * np.log10(4.37e-3 * f_max + 1)
#     erb_pts = np.linspace(erb_min, erb_max, num_filters)
#     f_centers = (10**(erb_pts / 21.4) - 1) / 4.37e-3
#     
#     # 2. Filter and calculate energies
#     energies = []
#     for fc in f_centers:
#         b, a = gammatone_iir_coefficients(fc, fs)
#         filtered = apply_iir_filter(b, a, signal)
#         energies.append(np.sum(filtered**2))
#         
#     # 3. Log-compression and DCT
#     log_energies = np.log(np.array(energies) + 1e-10)
#     # Manual DCT-II implementation
#     n = len(log_energies)
#     gtcc = np.zeros(num_ceps)
#     for i in range(num_ceps):
#         weights = np.cos(np.pi * i * (np.arange(n) + 0.5) / n)
#         gtcc[i] = np.sum(log_energies * weights)
#         
#     return gtcc
# 
# # Example Usage
# fs = 16000
# t = np.linspace(0, 0.5, fs // 2)
# signal = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
# coeffs = get_gtcc(signal, fs)
# print("GTCCs:", coeffs)

################################################################################

# import numpy as np
# 
# def design_gammatone_fir(fs, fc, n_taps=256, order=4):
#     """
#     Approximates a Gammatone FIR filter impulse response.
#     B is calculated based on Equivalent Rectangular Bandwidth (ERB).
#     """
#     t = np.arange(n_taps) / fs
#     erb = 24.7 * (4.37 * fc / 1000 + 1)
#     b = 1.019 * erb
#     # Gammatone impulse response: t^(n-1) * exp(-2*pi*b*t) * cos(2*pi*fc*t)
#     gt_ir = (t**(order-1)) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t)
#     return gt_ir / np.sum(np.abs(gt_ir))  # Gain normalization
# 
# def get_gtcc(signal, fs, n_filters=26, n_ceps=13, n_fft=512):
#     # 1. Pre-emphasis
#     signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
# 
#     # 2. Framing and Hamming Windowing
#     frame_len, frame_step = int(0.025 * fs), int(0.01 * fs)
#     num_frames = int(np.ceil(float(np.abs(len(signal) - frame_len)) / frame_step))
#     indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
#               np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
#     frames = np.pad(signal, (0, frame_len), 'constant')[indices.astype(np.int32)]
#     frames *= np.hamming(frame_len)
# 
#     # 3. Center Frequencies on the ERB Scale
#     erb_low = 21.4 * np.log10(0.00437 * 100 + 1)
#     erb_high = 21.4 * np.log10(0.00437 * (fs / 2) + 1)
#     f_centers = (10**(np.linspace(erb_low, erb_high, n_filters + 2) / 21.4) - 1) / 0.00437
# 
#     # 4. Filterbank Construction (FIR Response in Frequency Domain)
#     power_frames = np.abs(np.fft.rfft(frames, n_fft))**2 / n_fft
#     fb = np.zeros((n_filters, n_fft // 2 + 1))
#     for i in range(n_filters):
#         fir_coeff = design_gammatone_fir(fs, f_centers[i+1])
#         fb[i, :] = np.abs(np.fft.rfft(fir_coeff, n_fft))**2
# 
#     # 5. Log Filterbank Energies
#     features = np.log(np.dot(power_frames, fb.T) + 1e-10)
# 
#     # 6. Discrete Cosine Transform (DCT-II)
#     n = n_filters
#     dct_matrix = np.cos(np.pi * np.arange(n_ceps)[:, None] * (2 * np.arange(n) + 1) / (2 * n))
#     return np.dot(features, dct_matrix.T)
# 
# # Example usage
# fs = 16000
# test_signal = np.random.uniform(-1, 1, 16000)
# gtcc = get_gtcc(test_signal, fs)
# print(f"Extracted GTCC shape: {gtcc.shape}")