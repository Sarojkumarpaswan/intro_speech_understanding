import numpy as np

def voiced_excitation(duration, F0, Fs):
    '''
    Create voiced speech excitation.
    '''
    excitation = np.zeros(duration)
    period = int(np.round(Fs / F0))

    for n in range(0, duration, period):
        excitation[n] = -1.0

    return excitation


def resonator(x, F, BW, Fs):
    '''
    Generate the output of a resonator.
    '''
    N = len(x)
    y = np.zeros(N)

    r = np.exp(-np.pi * BW / Fs)
    theta = 2 * np.pi * F / Fs

    a1 = 2 * r * np.cos(theta)
    a2 = -r**2

    for n in range(2, N):
        y[n] = x[n] + a1 * y[n-1] + a2 * y[n-2]

    return y


def synthesize_vowel(duration, F0, F1, F2, F3, F4,
                     BW1, BW2, BW3, BW4, Fs):
    '''
    Synthesize a vowel.
    '''
    # Voiced excitation
    excitation = voiced_excitation(duration, F0, Fs)

    # Cascade of resonators (formants)
    y1 = resonator(excitation, F1, BW1, Fs)
    y2 = resonator(y1, F2, BW2, Fs)
    y3 = resonator(y2, F3, BW3, Fs)
    y4 = resonator(y3, F4, BW4, Fs)

    return y4
