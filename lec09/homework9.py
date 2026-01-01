import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    '''
    # Frame parameters
    frame_length = int(0.025 * Fs)  # 25 ms
    step = int(0.010 * Fs)          # 10 ms

    N = len(waveform)
    num_frames = 1 + (N - frame_length) // step

    energies = []
    frames = []

    for i in range(num_frames):
        start = i * step
        frame = waveform[start:start + frame_length]
        energy = np.sum(frame ** 2)
        energies.append(energy)
        frames.append(frame)

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    segments = []
    current_segment = []

    for i, energy in enumerate(energies):
        if energy > threshold:
            current_segment.append(frames[i])
        else:
            if current_segment:
                segments.append(np.concatenate(current_segment))
                current_segment = []

    # Catch last segment
    if current_segment:
        segments.append(np.concatenate(current_segment))

    return segments


def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment.
    '''
    models = []

    # Frame parameters
    frame_length = int(0.004 * Fs)  # 4 ms
    step = int(0.002 * Fs)          # 2 ms
    pre_emphasis = 0.97

    for segment in segments:
        # Pre-emphasis
        emphasized = np.append(segment[0], segment[1:] - pre_emphasis * segment[:-1])

        # Framing
        N = len(emphasized)
        num_frames = 1 + (N - frame_length) // step
        frames = np.zeros((num_frames, frame_length))

        for i in range(num_frames):
            start = i * step
            frames[i] = emphasized[start:start + frame_length]

        # FFT magnitude
        spectrum = np.abs(np.fft.fft(frames, axis=1))

        # Keep low-frequency half
        spectrum = spectrum[:, :frame_length // 2]

        # Convert to log spectrum
        spectrum = np.maximum(spectrum, 1e-10)
        log_spectrum = 20 * np.log10(spectrum)

        # Average across time
        model = np.mean(log_spectrum, axis=0)
        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Recognize speech using cosine similarity.
    '''
    # VAD on test speech
    test_segments = VAD(testspeech, Fs)

    # Convert test segments to models
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K))
    test_outputs = []

    for k, test_model in enumerate(test_models):
        for y, model in enumerate(models):
            # Cosine similarity
            numerator = np.dot(test_model, model)
            denominator = np.linalg.norm(test_model) * np.linalg.norm(model)
            sims[y, k] = numerator / denominator

        # Choose best matching label
        best_index = np.argmax(sims[:, k])
        test_outputs.append(labels[best_index])

    return sims, test_outputs
