import numpy as np
import librosa
from scipy.signal import lfilter

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    '''
    nframes = 1 + (len(speech) - frame_length) // frame_skip
    
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))
    
    for i in range(nframes):
        start = i * frame_skip
        end = start + frame_length
        frame = speech[start:end]

        # Compute LPC coefficients
        a = librosa.lpc(frame, order)
        A[i, :] = a

        # Compute excitation (residual)
        e = lfilter(a, [1.0], frame)
        excitation[i, :] = e

    return A, excitation


def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    '''
    nframes = A.shape[0]
    frame_length = e.shape[1]
    order = A.shape[1] - 1

    output_length = nframes * frame_skip + (frame_length - frame_skip)
    synthesis = np.zeros(output_length)

    for i in range(nframes):
        start = i * frame_skip
        end = start + frame_length

        # Filter excitation through LPC synthesis filter
        frame = lfilter([1.0], A[i], e[i])
        synthesis[start:end] += frame

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    '''
    Create robot voice excitation using impulse train.
    '''
    nframes, frame_length = excitation.shape
    
    gain = np.zeros(nframes)
    e_robot = np.zeros(nframes * frame_skip)

    for i in range(nframes):
        # Gain = RMS of valid part (last frame_skip samples)
        valid_part = excitation[i, -frame_skip:]
        gain[i] = np.sqrt(np.mean(valid_part ** 2))

        # Create impulse train for this frame
        frame_exc = np.zeros(frame_skip)
        frame_exc[::T0] = 1.0

        # Apply gain
        frame_exc *= gain[i]

        start = i * frame_skip
        end = start + frame_skip
        e_robot[start:end] = frame_exc

    return gain, e_robot
