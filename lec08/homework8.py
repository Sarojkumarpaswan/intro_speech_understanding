import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    '''
    N = len(waveform)
    num_frames = 1 + (N - frame_length) // step

    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start:start + frame_length]

    return frames


def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    '''
    # FFT along frame_length axis
    stft = np.fft.fft(frames, axis=1)

    # Magnitude
    mstft = np.abs(stft)

    return mstft


def mstft_to_spectrogram(mstft):
    '''
    Convert magnitude STFT to decibels with dynamic range limiting.
    '''
    # Avoid log of zero
    eps = 0.001 * np.max(mstft)
    mstft_safe = np.maximum(mstft, eps)

    # Convert to decibels
    spectrogram = 20 * np.log10(mstft_safe)

    # Limit dynamic range to 60 dB
    max_val = np.max(spectrogram)
    spectrogram = np.maximum(spectrogram, max_val - 60)

    return spectrogram
