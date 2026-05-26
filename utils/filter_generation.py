import sys, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import samplerate
import scipy.io.wavfile as wavfile
import tensorflow as tf
import librosa


import pdb

def design_gammatone_fir(fs, fc, n_taps=256, order=4):
    """
    Approximates a Gammatone FIR filter impulse response.
    B is calculated based on Equivalent Rectangular Bandwidth (ERB).
    """
    t = np.arange(n_taps) / fs
    erb = 24.7 * (4.37 * fc / 1000 + 1)
    b = 1.019 * erb

    gt_ir = (t**(order-1)) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t)
    gt_ir = gt_ir / np.sum(np.abs(gt_ir))  # Gain normalization
    fir_coeffs = np.fft.rfft(gt_ir)

    freqs = np.fft.rfftfreq(n_taps, 1 / fs)
    magnitude = np.abs(fir_coeffs)
    power = np.abs(fir_coeffs)**2
    phase_angle = np.angle(fir_coeffs)

    return freqs, magnitude, phase_angle

################################################################################

NUM_CEPSTRAL_FRAMES = tf.cast(52, dtype=tf.float32)
WINDOW_LENGTH = 0.128

wav_file = Path(os.getcwd()).joinpath("audio_16khz.wav")
sample_rate, audio = wavfile.read(wav_file)

pdb.set_trace()

audio = np.mean(audio[:41000], axis=1)

fs = 16000     # Sampling rate (Hz)
f_min, f_max = 50, fs / 2
NUM_FILTERS = 64

N_TAPS = 2048

# 1. Pre-emphasis
# audio = samplerate.resample(audio, fs / sample_rate, converter_type='sinc_best')
audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

audio_len = tf.cast(len(audio), dtype=tf.float32)
audio_sample_rate = tf.cast(fs, dtype=tf.float32)

win_hop_lower = (audio_len - audio_sample_rate * WINDOW_LENGTH) / (audio_sample_rate * NUM_CEPSTRAL_FRAMES)
win_hop_upper = (audio_len - audio_sample_rate * WINDOW_LENGTH) / (audio_sample_rate * (NUM_CEPSTRAL_FRAMES - tf.cast(1, dtype=tf.float32)))
win_hop_lower = tf.math.ceil(win_hop_lower * 10**5) / 10**5
win_hop_upper = tf.math.floor(win_hop_upper * 10**5) / 10**5
win_hop = tf.random.uniform(
    shape=[],
    minval=win_hop_lower, 
    maxval=win_hop_upper,
    dtype=tf.float32)

################################################################################

# 1. Framing and Hamming Windowing
frame_len, frame_step = int(WINDOW_LENGTH * fs), int(win_hop.numpy() * fs)
num_frames = int(np.floor(len(audio) - frame_len) / frame_step) + 1

stft = tf.signal.stft(
    audio,
    frame_length=2048,
    frame_step=frame_step,
    window_fn=tf.signal.hamming_window
)

spectrogram = tf.abs(stft)

################################################################################

# 2. MFCC Filterbank
mfcc_filterbank = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=NUM_FILTERS, 
    num_spectrogram_bins=spectrogram.shape[-1], 
    sample_rate=fs, 
    lower_edge_hertz=f_min, 
    upper_edge_hertz=f_max
)

mel_spectrogram = tf.tensordot(spectrogram, mfcc_filterbank, 1)

# # 3. Log Mel Spectrogram
log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
 
# # 4. Compute MFCCs
mfcc_coefficients = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

################################################################################

# 5. GTCC Filterbank
erb_min = 21.4 * np.log10(4.37e-3 * f_min + 1)
erb_max = 21.4 * np.log10(4.37e-3 * f_max + 1)
erb_pts = np.linspace(erb_min, erb_max, NUM_FILTERS)
f_centers = (10**(erb_pts / 21.4) - 1) / 4.37e-3

differences = []
gtcc_filterbank = np.zeros((len(f_centers), N_TAPS // 2 + 1))
for i, fc in enumerate(f_centers):

    fc = fc.astype(int) - .0001
    freqs, fir_magnitude, fir_phase_angle = design_gammatone_fir(fs, fc, n_taps=N_TAPS)
    numerator_coeffs, denominator_coeffs = signal.gammatone(fc, 'iir', fs=fs)
    _, iir_freq_response = signal.freqz(numerator_coeffs, denominator_coeffs, worN=(2 * np.pi) * freqs / fs)

    gtcc_filterbank[i,:] = fir_magnitude

    # iir_magnitudes = 20 * np.log10(abs(iir_freq_response))
    # fir_magnitudes = 20 * np.log10(fir_magnitude)
    # diff = np.abs(iir_magnitudes - fir_magnitudes)
    # rmse = np.sqrt(np.mean(diff)**2)
    # differences.append(rmse)

    # # Plotting
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # ax1.semilogx(freqs, iir_magnitudes, label="IIR")
    # ax1.semilogx(freqs, fir_magnitudes, label="FIR")
    # ax1.axvline(fc, color='green', linestyle='--', label=f'Center Freq: {fc}Hz')
    # ax1.set_title('Gammatone Filter Frequency Response')
    # ax1.set_xlabel('Frequency [Hz]')
    # ax1.set_ylabel('Amplitude [dB]')

    # ax2.plot(freqs, np.angle(iir_freq_response), label="IIR")
    # ax2.plot(freqs, fir_phase_angle, label="FIR")

    # plt.grid(which='both', axis='both')
    # plt.legend()
    # plt.show()

gtcc_filterbank = tf.convert_to_tensor(gtcc_filterbank, dtype=tf.float32)
gammatone_spectrogram = tf.tensordot(spectrogram, tf.transpose(gtcc_filterbank), 1)
log_gammatone_spectrogram = tf.math.log(gammatone_spectrogram + 1e-10)

gtcc_coefficients = tf.signal.dct(log_gammatone_spectrogram, type=2, norm='ortho')

pdb.set_trace()

################################################################################

# indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
#           np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
# 
# frames = np.pad(audio, (0, frame_len), 'constant')[indices.astype(np.int32)]
# frames *= np.hamming(frame_len)
# power_frames = np.abs(np.fft.rfft(frames))**2

# n = gtcc_filterbank.shape[0]
# dct_matrix = np.cos(np.pi * np.arange(NUM_CEPSTRAL_FRAMES)[:, None] * (2 * np.arange(n) + 1) / (2 * n))
# features = np.log(np.dot(power_frames, filterbank.T) + 1e-10)

# plt.hist(differences, bins=30, edgecolor='black')
# plt.show()