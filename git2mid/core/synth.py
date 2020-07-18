from typing import Dict, List, Any

import numpy as np
import librosa
import numpy as np

# sampling information
Fs = 44100  # sample rate
T = 1 / Fs  # sampling period

# signal information
freq = 100  # in hertz, the desired natural frequency


def generate_note(freq, length=5, string='E', noise_harmonics_level=False, noise_harmonics_freq=False):
    """ this is a docstring
    :param freq: frequency in Hz
    :param length: of the chord in seconds
    :param string: one of ['E', 'a', 'd', ...]
    :param noise_harmonics_level:
    :param noise_harmonics_freq:
    :return:
    """
    # harm_level = [1, 1, 0.8, 0.7, 0.6, 0.5, 0.4]
    harm_dict: Dict[Any, List[int]] = dict(
        E=[10, 13, 11, 16, 12, 9, 8],
        A=[15, 14, 25, 17, 10, 7, 3],
        D=[19, 21, 18, 14, 9, 6, 3],
        G=[18, 14, 14, 14, 9, 6, 4],
        H=[15, 35, 24, 21, 18, 9, 6],
        e=[18, 34, 18, 10, 11, 9, 6],
    )

    harm_level = harm_dict[string]
    harm_level /= np.sum(harm_level)

    t = length + 1
    N = Fs * t  # total points in signal

    for i, level in enumerate(harm_level):

        if noise_harmonics_level:
            level += np.random.randn(1) * 0.005  # add somie nosie to harmonics levels
        if noise_harmonics_freq:
            harm_freq_noises = [0, 0, 0.01, 0.05, 0.1, 0.1]
            freq_harm = freq  # + np.random.randn(1) * harm_freq_noises[i]

        omega = 2 * np.pi * (freq * i)  # angular frequency for sine waves

        t_vec = np.arange(N) * T  # time vector for plotting

        random_start = np.random.randint(Fs)
        random_end = random_start + length * Fs

        y = np.sin(omega * t_vec) * level
        y = y[random_start:random_end]

        if i == 0:
            note = y
        else:
            note += y

    return librosa.util.normalize(note)


def generate_chord(detune=True, detune_strength=3):
    """
    :param detune:
    :param detune_strength:
    :return:
    """
    freqs_ideal = librosa.midi_to_hz([40, 45, 50, 55, 59, 64])

    # TODO: detune in cent
    detune_random = np.random.rand(1)

    if detune and detune_random < 1:
        detune_diff = np.random.randn(6) * detune_strength
        #         # leave somestrings tuned with a 50% chance
        #         if np.random.rand(1) > 0.5:
        #             detune_diff_randomize = np.random.randint(2, size=6)
        #             detune_diff *= detune_diff_randomize
        #             detune_diff += np.random.rand(6)*0.001

        freqs = freqs_ideal - detune_diff  # was 3
    else:
        freqs = freqs_ideal  # - (np.random.randn(6))#*0.01)
        # print('not detuned!')

    chord = []

    for i, string in enumerate(['E', 'A', 'D', 'G', 'H', 'e']):
        note = generate_note(freqs[i], length=2, string=string, noise_harmonics_level=True)
        note *= np.random.rand()
        chord.append(note)

    chord = np.sum(chord, axis=0)

    fade = (np.arange(chord.size) / 1)[::-1]

    return chord * fade, freqs


def fft_chord(chord):
    """

    :type chord: object
    """
    # TODO: window function
    N = chord.size
    Y_k = np.fft.fft(chord, norm='ortho')[0:int(N / 2)] / N  # FFT function from numpy
    Y_k[1:] = 2 * Y_k[1:]  # need to take the single-sided spectrum only
    spectrum = np.abs(Y_k)  # be sure to get rid of imaginary part

    return spectrum


def generate_data(n, label='diff', detune=10):
    """

    :type n: object
    """
    import mir_eval

    X = []
    X_chords = []
    Y = []

    for i in range(n):

        chord, freqs = generate_chord(detune_strength=detune)
        # print(freqs)
        chord /= np.max(abs(chord))
#         fft = np.mean(librosa.constantq.cqt(chord, n_bins=84, bins_per_octave=12, sr=16000), axis=1)
#         fft = librosa.util.normalize(fft)
        #print('Basst.')
        y = freqs

        if label == 'diff' or label == 'sign' or label == 'three':
            freqs_ideal = librosa.midi_to_hz([40, 45, 50, 55, 59, 64])
            # print(freqs_ideal)
            freqs_ideal = mir_eval.melody.hz2cents(freqs_ideal)
            freqs = mir_eval.melody.hz2cents(freqs)
            diff = freqs_ideal - freqs
            y = diff

        if label == 'sign':
            y = np.sign(y)
            y[y == -1] = 0

        if label == 'three':
            y = np.sign(y)
        #             y[y==0] = 0.5
        #             y[y==-1] = 0

        #X.append(fft)
        X_chords.append(chord)
        Y.append(y)

    X = np.array(X)
    X_chords = np.array(X_chords)
    Y = np.array(Y)

    return X, X_chords, Y

def crop_to_min(arrays):
    
    min_len = min([array.shape[0] for array in arrays])
    arrays = [array[:min_len] for array in arrays]
    
    return arrays


def resynth(out, notes=37, stretch=1, add_wav=None, harmonizer=[0], add_wav_offset=0, synth_offset=0):
    
    from scipy import signal
    
    """Takes many hot and resynthesizes it"""
    
    # out.shape
    # >>> (48, 128)
    
    wav_out = np.zeros(out.shape[1]*64*stretch)
    print(wav_out.shape)
    
    for j in harmonizer:

        for i, note in enumerate(out):

            if i < 5:
                string='E'
            elif i < 10:
                string='A'
            elif i < 15:
                string='D'
            elif i < 20:
                string='G'
            elif i < 25:
                string='H'
            else:
                string='e'

            mod = signal.resample(note, note.shape[0]*64)
            wav = generate_note(librosa.midi_to_hz(i+40+j), length=(wav_out.shape[0]//44100)+1, string=string)

            wav, mod, wav_out = crop_to_min([wav, mod, wav_out])
            wav *= mod
            wav_out += wav

        wav_out /= notes
    
    if add_wav is not None:
        
        
        add_wav = np.pad(add_wav, (add_wav_offset, 0))
        wav_out = np.pad(wav_out, (synth_offset, 0))
        
        max_len = min(add_wav.shape[0], wav_out.shape[0])
        
        add_wav = add_wav[:max_len]
        wav_out = wav_out[:max_len]

    
            

        #wav_out = np.vstack((wav_out.ravel(), add_wav.ravel()))
        
        return wav_out, add_wav
    
    else:
        wav_out /= np.max(abs(wav_out))
    

    
        return wav_out