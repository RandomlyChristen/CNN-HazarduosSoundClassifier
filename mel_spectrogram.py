import csv
import decimal
import math
import os

import numpy as np
import scipy.io.wavfile as wav


def magspec(frames, NFFT):
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def powspec(frames, NFFT):
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    win = winfunc(frame_len)
    frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    return frames * win


def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def calculate_nfft(samplerate, winlen):
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft


def fbank(signal, samplerate, winlen, winstep,
          nfilt, nfft, lowfreq, highfreq, preemph,
          winfunc):
    highfreq = highfreq or samplerate / 2
    signal = preemphasis(signal, preemph)
    frames = framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = powspec(frames, nfft)
    energy = np.sum(pspec, 1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = np.dot(pspec, fb.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    return feat, energy


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank_ = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank_[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank_[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank_


# Convert Frequency to Mel Scale
def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.)


# Convert Mel Scale to Frequency Mel Scale은 Filter Bank를 나눌때 어떤 간격으로 나누어야 하는지 알려주며, 간격을 나누는 방법은 아래와 같다
def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def mel_spectrogram(signal, samplerate=44100):
    nfft = calculate_nfft(samplerate, 0.025)
    feat, energy = fbank(signal, samplerate, 0.025, 0.01, 90, nfft, 0, None, 0.97, lambda x: np.ones((x,)))
    feat = np.log(feat)
    return feat


if __name__ == '__main__':
    LABELED_DATA_PATH = 'data/'
    RESULT_CSV_PATH = 'data/mel_spectrogram.csv'

    csv_list = []

    for label in os.listdir(LABELED_DATA_PATH):
        if not os.path.isdir(LABELED_DATA_PATH + label):
            continue

        for wav_filename in os.listdir(LABELED_DATA_PATH + label):
            print('label', label)
            if not wav_filename.endswith('.wav'):
                continue
            sample_rate, data = wav.read(LABELED_DATA_PATH + label + '/' + wav_filename)
            mel_spec = mel_spectrogram(data, sample_rate)
            spec_array = np.ravel(mel_spec)
            spec_array = np.append(spec_array, (1 if label == 'danger' else 0)).tolist()

            csv_list.append(spec_array)

    with open(RESULT_CSV_PATH, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)
