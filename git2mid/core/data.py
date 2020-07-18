import math
import os
import pickle
import random

import librosa
import numpy as np
import pandas as pd
import scipy.stats as stats
import textdistance
from pysndfx import AudioEffectsChain

import config


class LabelCoderDict:

    def __init__(self):
        self.chords_classes = pd.read_json(os.path.join(config.GIT2MID_CHORD_ROOT, "chord_classes.json"))
        self.n_classes = len(self.chords_classes['class'].unique()) + 1
        self.chords_classes['class'] = self.chords_classes['class'] / self.chords_classes['class'].max() * (
                self.n_classes - 1)


# self.chords_classes['one_hot'] = [to_categorical(chord, num_classes=self.n_classes) for chord in
# self.chords_classes['class']]

#         self.midi_to_one_hot_dict = dict(zip([tuple(chord) for chord in self.chords_classes['midi_notes']],
#                                                 self.chords_classes['one_hot'])) 
#         self.one_hot_to_midi_dict = dict(zip([np.argmax(chord) for chord in self.chords_classes['one_hot']],
#                                                 self.chords_classes['midi_notes']))


class LabelCoder:
    def __init__(self, LabelCoderDict):

        #         self.midi_to_one_hot_dict = LabelCoderDict.midi_to_one_hot_dict
        #         self.one_hot_to_midi_dict = LabelCoderDict.one_hot_to_midi_dict
        #         self.n_classes = LabelCoderDict.n_classes
        self.temp = []

    def binarize_many_hot(self, many_hot, threshold=0):
        many_hot = many_hot.copy()
        many_hot[many_hot > 0] = 1
        return many_hot

    def midi_to_one_hot(self, midi):
        return self.midi_to_one_hot_dict[tuple(midi)]

    def one_hot_to_midi(self, one_hot, top=0):
        arg = np.argsort(one_hot)[::-1][top]
        return self.one_hot_to_midi_dict[arg]

    def one_hot_to_midi_(self, one_hot, top=0):
        return [self.one_hot_to_midi(step, top) for step in one_hot]

    def midi_to_low(self, midi):
        return self.midi_to_low_dict[tuple(midi)]

    def midi_to_many_hot(self, midi, note_min=40, note_range=12 * 4):
        many_hot = np.zeros(note_range)
        if midi != []:
            many_hot[np.array(midi) - note_min] = 1
        return many_hot

    def low_to_midi(self, low):
        return np.argmax(low, axis=1) + 40

    def many_hot_to_one_hot(self, many_hot):
        many_hot = many_hot.copy()
        midi = self.many_hot_to_midi(many_hot)
        return self.midi_to_one_hot(midi)

    def many_hot_to_one_hot_(self, many_hot):
        return np.stack([self.many_hot_to_one_hot(step) for step in many_hot])

    def many_hot_to_low(self, many_hot):
        many_hot = many_hot.copy()
        midi = self.many_hot_to_midi(many_hot)
        low = np.zeros(37)
        low[np.min(midi) - 40] = 1
        return low

    def many_hot_to_low_(self, many_hot):
        return np.stack([self.many_hot_to_low(step) for step in many_hot])

    def many_hot_to_midi(self, many_hot: object, threshold: object = 0, binarize: object = True) -> object:
        many_hot = many_hot.copy()
        if binarize:
            many_hot = self.binarize_many_hot(many_hot)
        condition = many_hot > threshold
        return np.where(condition)[0] + 40

    def round_one_hot(self, one_hot):
        one_hot = np.array(one_hot)
        arg_max = np.argmax(one_hot)
        one_hot[arg_max] = 1
        one_hot[one_hot < 1] = 0
        return list(one_hot)

    def round_one_hot_(self, one_hot):
        return [round_one_hot(step) for step in one_hot]


def blur_attack_label(piano, width=200):
    original_length = piano.shape[0]
    piano = np.pad(piano, original_length)
    mu = 0
    variance = 1

    sigma = math.sqrt(variance)
    n = np.linspace(mu - 3 * sigma, mu + 3 * sigma, width)
    n = stats.norm.pdf(n, mu, sigma)
    attack = piano.argmax()
    piano[attack - int(width / 2):attack + int(width / 2)] = n
    piano /= np.max(piano)

    return piano[original_length:-original_length]


def sample_sequences(X, Y, n=10000, timesteps_features=100, return_sequences=True, binary_labels=True):
    '''100 timesteps_features = 2.27 ms @ 44100 Hz'''

    X_ = []
    Y_ = []

    for i in range(n):

        sample = np.random.randint(X.shape[0])
        timestep = np.random.randint(X.shape[1] - timesteps_features)

        X_.append(X[sample:sample + 1, timestep:timestep + timesteps_features, :])

        if return_sequences:
            Y_.append(Y[sample:sample + 1, timestep:timestep + timesteps_features])
        else:
            Y_.append(Y[sample:sample + 1, timestep:timestep + timesteps_features][:, -1])

    X_ = np.vstack(X_)
    Y_ = np.vstack(Y_)

    if binary_labels:
        Y_[Y_ > 0] = 1

    return X_, Y_


def sample_sequences_(X, Y, n=10000, timesteps_features=100, return_sequences=True, binary_labels=True):
    '''100 timesteps_features = 2.27 ms @ 44100 Hz'''
    '''this is for samples foro one auduio file, i.e. n_dims = 2, (timesteps, features) -> for samples generated with midi-pu'''

    X_ = []
    Y_ = []

    for i in range(n):

        timestep = np.random.randint(X.shape[0] - timesteps_features)

        X_.append(X[np.newaxis, timestep:timestep + timesteps_features, :])

        if return_sequences:
            Y_.append(Y[np.newaxis, timestep:timestep + timesteps_features])
        else:
            Y_.append(Y[np.newaxis, timestep:timestep + timesteps_features])

    X_ = np.vstack(X_)
    Y_ = np.vstack(Y_)

    if binary_labels:
        Y_[Y_ > 0] = 1

    return X_, Y_


def filter_lowpass(audio, range=(0.01, 0.2)):
    from scipy.signal import lfilter, lfilter_zi, filtfilt, butter

    low = np.random.uniform(range[0], range[1])
    # high = np.random.uniform(0.2, 0.99)

    xn = audio
    # Create an order 3 lowpass butterworth filter.
    b, a = butter(3, low, btype='low')

    # Apply the filter to xn.  Use lfilter_zi to choose the initial condition
    # of the filter.
    zi = lfilter_zi(b, a)
    z, _ = lfilter(b, a, xn, zi=zi * xn[0])

    # Apply the filter again, to have a result filtered at an order
    # the same as filtfilt.
    z2, _ = lfilter(b, a, z, zi=zi * z[0])

    # Use filtfilt to apply the filter.
    y = filtfilt(b, a, xn)

    return y


def hz_to_cents(freq, baseFreq=10):
    LOG10_TWO = 0.301029995664
    return 1200.0 * (np.log10(freq / baseFreq) / LOG10_TWO)


def cents_to_hz(cents, baseFreq=10):
    return baseFreq * np.power(2.0, (cents / 1200.0))


def hz_to_speed(in_hz, out_hz):
    return out_hz / in_hz


def speedx(sound_array, factor, resample=8):
    if resample != False:
        sound_array = librosa.resample(sound_array, 44100, 44100 * resample, res_type='kaiser_fast')
    indices = np.round(np.arange(0, len(sound_array), factor))
    indices = indices[indices < len(sound_array)].astype(int)
    sound_array = sound_array[indices.astype(int)]
    if resample != False:
        sound_array = librosa.resample(sound_array, 44100 * resample, 44100, res_type='kaiser_fast')

    return sound_array


def pyrubber(audio, pitch_hz, pitch_ref_hz, mode=5, formant=True):
    x = pitch_ref_hz / pitch_hz
    print('freq ratio', x)

    librosa.output.write_wav('temp/temp.wav', audio, 44100)

    command = 'rubberband temp/temp.wav temp/temp_rubbered.wav -f {} -c {}'.format(x, mode)
    print(command)
    if formant:
        comand = command + ' -F'
    os.system(command)

    audio_ = librosa.load('temp/temp_rubbered.wav', sr=44100)[0]

    return audio_


def pitch_correction(audio, pitch, pitch_ref, lib='rubberband'):
    correction = hz_to_cents(pitch_ref) - hz_to_cents(pitch)
    correction_semitones = correction / 100
    # print(correction_semitones)

    audio = np.asfortranarray(audio)

    if lib == 'librosa':

        audio = librosa.effects.pitch_shift(
            audio,
            44100,
            correction_semitones,
            res_type='kaiser_fast'
        )



    elif lib == 'rubberband':

        mode = np.random.randint(7)
        audio = pyrubber(audio, pitch, pitch_ref)

    else:
        audio = speedx(audio, hz_to_speed(pitch, pitch_ref))

    return audio


def note_to_chroma(note):
    return note[:-1]


def notes_to_chroma(notes):
    return list(set([note_to_chroma(note) for note in notes]))


class FishmanGuitarDatabase():

    def __init__(self,
                 path= os.path.join(config.GIT2MID_DB_ROOT, 'fishman/single_notes/single_notes.pkl'),
                 sr=44100,
                 preload_audio=False,
                 ):

        '''
        There are three main components in this init
        
        1. data frame | pointing to the audio files and holding metadata
        2. chord data | base that is used for generating the chordsamples
        3. the actual audiofiles | when preload_audio = True a dict will be generated
        
        '''
        import pandas as pd

        self.df = pd.read_pickle(path)

        # clean with blacklist
        self.blacklist = pd.read_table(os.path.join(config.GIT2MID_DB_ROOT, 'fishman/single_notes/blacklist.txt'), header=None,
                                       index_col=0).index
        self.df = self.df.drop(self.blacklist)

        # change the file paths in the df
        def generate_new_path(old_path, path):
            # print(old_path)
            new_root = '/'.join(path.split('/')[:-1])
            filename = '/'.join(old_path.split('/')[-2:])
            new_path = '/'.join((new_root, filename))
            return new_path

        self.df.index = [generate_new_path(x, path) for x in self.df.index]

        # chord shapes to use for the chords 
        self.chords = pd.read_json('data/chords/chord_classes.json')
        self.chords['poly'] = [len(x) for x in self.chords.midi_notes]
        self.top_chords = pd.read_json('data/chords/top_chords.json')
        self.top_chords['poly'] = [len(x) for x in self.top_chords.midi_notes]

        self.top_chords['chroma'] = [notes_to_chroma(note) for note in librosa.midi_to_note(self.top_chords.midi_notes)]

        self.sr = sr
        self.preload_audio = preload_audio
        if self.preload_audio:
            self.audio_dict = {}
            for file in self.df.index:
                self.audio_dict[file] = librosa.load(file, sr=44100)[0]

        self.verbose = False

    def padding(self, audio, mode='zeros', pad=(0, 0), noise_range=(0, 1e-3), noise_floor=False):
        if mode == 'zeros':
            audio = np.pad(audio, pad)
        if mode == 'noise':
            noise_level = np.random.uniform(noise_range[0], noise_range[1])
            if noise_floor:
                audio += np.random.randn(audio.shape[0]) * noise_level
            audio = np.hstack([
                np.random.randn(pad[0]) * noise_level,
                audio,
                np.random.randn(pad[1]) * noise_level,
            ])

        return audio

    def get_file(self, note=40, split='train', return_pitch=False):

        f0_error_tolerance = 50  # cent

        df = self.df[self.df.f0_error_cent < f0_error_tolerance]

        if split == 'all':
            df = df[df.note == note].sample(1)
        else:
            df = df[(df.note == note) & (df.split == split)].sample(1)
        file = df.index[0]
        offset = int((df.pre_roll_ms.values[0] / 1000) * self.sr)  # convert to samples
        f0 = df.f0_mean
        f0_error = df.f0_error_cent.values
        if self.verbose:
            print('mean', df.f0_mean.values, 'median', df.f0_median.values, 'std', df.f0_std.values, 'error', f0_error)

        if return_pitch:
            return file, offset, f0
        else:
            return file, offset

    def get_audio(self, file, duration=None, offset=0, normalize=True, augment=False, augment_prob=0.5):
        '''
        duration (samples)
        offset (samples)
        '''
        if self.verbose: print('file', file)
        if self.preload_audio:
            audio = self.audio_dict[file][offset:duration]
        else:
            if duration != None:
                audio = librosa.load(file, sr=self.sr, duration=(duration / self.sr))[0]
            else:
                audio = librosa.load(file, sr=self.sr)[0]

        if augment:
            if np.random.rand(1) > augment_prob:
                audio = filter_lowpass(audio)
        if normalize:
            audio /= np.max(abs(audio))
        return audio

    def get_chord(self, poly='random', top_k=None):
        for i in range(10):
            try:
                if poly == 'random':
                    poly = np.random.randint(1, 7)
                if poly in [1, 2]:
                    notes = list(np.random.randint(40, self.chords.midi_notes.values.max()[0] + 1, size=poly))
                else:
                    chord_df = random.choice([self.chords, self.top_chords.iloc[:top_k]])
                    df = chord_df[chord_df.poly == poly].sample(1)
                    notes = df.midi_notes.values[0]
                if self.verbose:
                    print(poly)
                    print('notes', notes, librosa.midi_to_note(notes))
                    try:
                        print('TAB:', df.joint.values)
                    except:
                        pass
                break
            except Exception as ex:
                print(ex, 'Retry', i + 1)

        return notes

    def get_note(self,
                 note,
                 split='train',
                 pre_roll=None,
                 label_pre_roll=0,
                 sustain=None,
                 samples=8192 * 2,
                 attack=8192,
                 padding_mode='zeros',
                 noise_floor=0,
                 normalize=True,
                 augment=False,
                 correct_pitch_prob=0,
                 augment_pitch=False,
                 pitch_range=10,
                 pitch_lib=None
                 ):

        if np.random.rand(1) <= correct_pitch_prob:

            note_offset = np.random.randint(3)  # nehmen falsche note die nachher auf richtigen Pitch wird
            if self.verbose: print('note_offset', note_offset)
            file, offset, f0 = self.get_file(
                np.max((note + note_offset, 40)),
                split=split,
                return_pitch=True
            )

            audio = self.get_audio(file, normalize=normalize, augment=augment)

            if augment_pitch:
                random_cents = np.random.randint(-pitch_range, pitch_range)
                f0_target = cents_to_hz(hz_to_cents(librosa.midi_to_hz(note)) + random_cents)
            else:
                f0_target = librosa.midi_to_hz(note)

            f0 = f0.values[0]
            audio = pitch_correction(audio, f0, f0_target, lib=pitch_lib)

            if self.verbose:
                print('corrected pitch from {} to {}'.format(f0, f0_target))
                if augment_pitch:
                    print('augmented pitch from {} to {}, meaning {} cents'.format(f0, f0_target, random_cents))

        else:

            file, offset, f0 = self.get_file(
                note,
                split=split,
                return_pitch=True
            )

            file, offset = self.get_file(note, split=split)
            audio = self.get_audio(file, normalize=normalize, augment=augment)

        # some checks
        if pre_roll > attack:
            pre_roll = attack
            # print('Warning: Pre_roll > attack. Use max.')
        if offset - pre_roll < 0:
            pre_roll = offset
            # print('Warning: Pre_roll of audio file too short. Use max.')
        if sustain > samples - attack:
            sustain = samples - attack
            # print('Warning: Sustain too long for samples. Use max.')

        pre_attack = audio[:offset][offset - pre_roll:]
        post_attack = audio[offset:][:sustain]

        # [------space_pre_attack--------att|ack-------space_post_attack------]
        space_pre_attack = attack
        space_post_attack = samples - attack

        # checks
        pre_attack_too_short = len(pre_attack) < space_pre_attack
        pre_attack_too_long = len(pre_attack) > space_pre_attack
        post_attack_too_short = len(post_attack) < space_post_attack
        post_attack_too_long = len(post_attack) > space_post_attack

        # doc zero paddings
        zero_padding_beginning = 0
        zero_padding_end = 0

        if pre_attack_too_short:
            how_much = space_pre_attack - len(pre_attack)
            pre_attack = self.padding(pre_attack, pad=(how_much, 0), mode=padding_mode)
            zero_padding_beginning = how_much
        if pre_attack_too_long:
            how_much = len(pre_attack) - space_pre_attack
            pre_attack = pre_attack[how_much:]
        if post_attack_too_short:
            how_much = space_post_attack - len(post_attack)
            post_attack = self.padding(post_attack, pad=(0, how_much), mode=padding_mode)
            zero_padding_end = how_much
        if post_attack_too_long:
            how_much = len(post_attack) - space_post_attack
            post_attack = post_attack[:-how_much]

        out = np.zeros(samples) + (np.random.randn(samples) * noise_floor)
        out[:attack] += pre_attack
        out[attack:] += post_attack

        piano_roll = np.zeros(samples)
        if zero_padding_end > 0:
            piano_roll[attack:-zero_padding_end] = 1
        else:
            piano_roll[attack:] = 1
        piano_roll[attack - label_pre_roll:attack] = 1

        return out, pre_attack, post_attack, piano_roll

    def sample_chord(self,
                     notes=None,
                     split='train',
                     use_sequence=True,
                     poly='random',
                     notes_range=48,
                     samples=8192,
                     sr=44100,
                     amplitude='random',
                     amplitude_min=0.1,
                     amplitude_max=1,
                     attack='random',  # delay == attack
                     attack_min=0,
                     attack_max=8192,
                     pre_roll='random',
                     pre_roll_min=0,
                     pre_roll_max=4096,
                     label_pre_roll='follow',
                     label_pre_roll_min=0,
                     label_pre_roll_max=4096,
                     negative_delay_labels=0,
                     use_peak=False,
                     attacks_only=False,
                     attacks_sustain=False,
                     blur_attacks=False,
                     blur_width=200,
                     padding_mode='noise',
                     noise_floor=0.0,
                     noise_labels=0.0,
                     sustain='random',
                     sustain_min=64,
                     sustain_max=8192,
                     noise_prob=0,
                     augment=False,
                     correct_pitch_prob=0,
                     pitch_range=10,
                     augment_pitch=True,
                     pitch_lib=None
                     ):

        if notes is None:
            notes = self.get_chord(poly=poly)
        if self.verbose: print(notes)
        poly = len(notes)

        # prepare empty outputs
        output = np.zeros((poly, samples))
        piano_roll = np.zeros((notes_range, int(samples / 64)))
        piano_roll_energy = np.zeros((notes_range, int(samples / 64)))

        # get all single notes audios for chord
        for i, note in enumerate(notes):
            # randomize TODO: wrap this up!
            if attack == 'random':
                attack_ = np.random.randint(attack_min, attack_max + 1)
            else:
                attack_ = attack

            if pre_roll == 'random':
                pre_roll_ = np.random.randint(pre_roll_min, pre_roll_max + 1)
            else:
                pre_roll_ = pre_roll

            if sustain == 'random':
                sustain_ = np.random.randint(sustain_min, sustain_max + 1)
            else:
                sustain_ = sustain

            if label_pre_roll == 'random':
                label_pre_roll_ = np.random.randint(label_pre_roll_min, label_pre_roll_max + 1)
            else:
                label_pre_roll_ = label_pre_roll

            if label_pre_roll == 'follow':
                label_pre_roll_ = pre_roll_

            if amplitude == 'random':
                amplitude_ = np.random.uniform(amplitude_min, amplitude_max)

            audio, _, _, piano = self.get_note(
                note,
                split=split,
                pre_roll=pre_roll_,
                label_pre_roll=label_pre_roll_,
                sustain=sustain_,
                samples=samples,
                attack=attack_,
                padding_mode=padding_mode,
                noise_floor=noise_floor,
                normalize=True,
                correct_pitch_prob=correct_pitch_prob,
                augment=augment,
                augment_pitch=augment_pitch,
                pitch_range=pitch_range,
                pitch_lib=pitch_lib
            )

            output[i] = audio * amplitude_

            if attacks_only:
                piano = np.zeros(piano.shape)
                if attacks_sustain:
                    piano[attack_:] = 1
                else:
                    piano[attack_] = 1
                if blur_attacks:
                    piano = blur_attack_label(piano, width=blur_width)

            # piano_roll[note-notes_range] = resample(piano, int(samples/64))[:int(samples/64)] # makes weird dotty things
            piano = piano[::64][:int(samples / 64)]
            piano /= np.max(piano)
            piano_roll[note - 40] = piano

            # resampling makes some negative values sometimes

            if use_peak == False:
                energy = librosa.feature.rms(audio, hop_length=64, center=True)[:, :int(samples / 64)].ravel()
            else:
                energy = np.stack([max(x) for x in abs(librosa.util.frame(np.pad(audio, (1024, 0)), 1024, 64)).T])[
                         :int(samples / 64)].ravel()

            if use_sequence:
                if negative_delay_labels > 0:
                    actual_end_value = energy[-1]
                    energy = np.roll(energy, -int(negative_delay_labels / 64))
                    # TODO: be more precise, extrapolate etc
                    energy[-int(negative_delay_labels / 64):] = actual_end_value

            piano_roll_energy[note - 40] = energy

        audio = np.sum(output, axis=0) / poly

        # norm and scale
        amplitude_max_final = np.random.uniform(amplitude_min, amplitude_max)

        audio /= np.max(abs(audio))
        audio *= amplitude_max_final

        piano_roll_energy += (np.random.rand(piano_roll_energy.shape[0],
                                             piano_roll_energy.shape[1])
                              * noise_labels)
        piano_roll_energy /= np.max(piano_roll_energy)
        piano_roll_energy *= amplitude_max_final

        piano_roll[piano_roll < 0] = 0

        # output only noise sometimes
        #         if np.random.rand(1) < noise_prob:
        #             print('Noise')
        #             audio = np.random.randn(samples)*np.random.rand(1)
        #             audio /= np.max(abs(audio))
        #             piano_roll = np.zeros((notes_range, samples))
        #             piano_roll_energy = piano_roll.copy()

        if use_sequence:

            return audio, piano_roll, piano_roll_energy

        else:

            return audio, notes

    ############################################### post processing ######################################################################################################################

    def get_winner(self, notes):
        top_chords = self.top_chords.copy()
        query = notes_to_chroma(librosa.midi_to_note(notes))
        top_chords['chroma_dist'] = np.array(
            [textdistance.levenshtein(query, chord) for chord in top_chords.chroma.values])
        top_chords['dist'] = np.array([textdistance.levenshtein(notes, chord) for chord in top_chords.midi_notes])
        top_chords['chroma_dist'] = top_chords.chroma_dist.max() - top_chords.chroma_dist
        # top_chords['dist'] = np.array([len(textdistance.lcsseq(query, chord)) for chord in top_chords.chroma.values])
        candidates = top_chords[top_chords.dist == top_chords.dist.min()].sort_values(['dist', 'chroma_dist'],
                                                                                      ascending=False)
        winner = candidates.iloc[0]
        return winner.midi_notes


######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################


################################################################################# LEGACY ##############################################################################################


class GuitarDatabase():

    def __init__(self, dbs):

        fender, ibanez, ibanez_2, ibanez_3 = pickle.load(open("guitar_dbs.pkl", "rb"))

        dbs_dict = {
            'fender': fender,
            'ibanez': ibanez,
            'ibanez_2': ibanez_2,
            'ibanez_3': ibanez_3

        }

        self.dbs = [dbs_dict[db] for db in dbs]

        self.chords = pd.read_json('chord_classes.json')
        self.chords['poly'] = [len(x) for x in self.chords.midi_notes]

    def sample_chord(self,
                     notes=None,
                     samples=44100,
                     sr=44100,
                     res_type=None,
                     noise=True,
                     arpeggio_prob=0,  #
                     random_crop=False,
                     return_audio=False,
                     extract_features=False):

        run = True

        while run == True:

            try:
                import random

                audio = np.zeros(samples)

                delay = 0  # np.random.randint(samples)

                if notes == None:
                    notes = self.chords.sample(1).midi_notes.values[0]

                for note in notes:
                    # print(note, delay)
                    db = random.choice(self.dbs)
                    single_note_audio = db[note - 40]

                    # make start a zero crossing
                    start = np.min(np.where(librosa.zero_crossings(single_note_audio) == True))
                    single_note_audio = single_note_audio[start:]

                    if single_note_audio.shape[0] < samples:
                        single_note_audio = np.pad(single_note_audio, (0, samples - single_note_audio.shape[0]),
                                                   mode='constant')
                        assert single_note_audio.shape[0] == samples

                    # data augmentation 
                    fx = (
                        AudioEffectsChain()
                            .bandpass(frequency=np.random.randint(100, 1000), q=np.random.randint(1, 3))
                    )

                    single_note_audio = fx(single_note_audio)
                    single_note_audio *= np.random.uniform(0.5, 1)

                    # add

                    if random_crop:

                        zero_crossing = librosa.zero_crossings(single_note_audio)

                        random_end = np.random.randint(samples / 10, samples)
                        random_fade = np.random.randint(random_end + 1000, random_end + 10000)
                        while zero_crossing[random_end] == False and zero_crossing[random_fade] == False:
                            random_end = np.random.randint(samples / 10, samples)
                            random_fade = np.random.randint(random_end + 1000, random_end + 10000)

                        single_note_audio[random_end:random_end + random_fade] *= np.linspace(1, 0, random_fade)
                        single_note_audio[random_end:] = 0

                    audio[delay:] += single_note_audio[:samples - delay]

                    if np.random.rand(1) < arpeggio_prob:
                        arpeggio = True
                    else:
                        arpeggio = False
                    if arpeggio:
                        if np.random.rand(1) > 0.5:
                            delay += np.random.randint(samples)
                        else:
                            delay += np.random.randint(100)
                        if delay > samples - 1000: delay = 0

                if noise and np.random.randint(2) == 1:
                    audio += (np.random.normal(size=audio.shape) * np.random.uniform(0, 0.005))

                if extract_features:
                    # extract features
                    features = abs(librosa.cqt(
                        audio,
                        sr,
                        n_bins=84 * 10,
                        bins_per_octave=12 * 10,
                        hop_length=64,
                        res_type=res_type
                    ))

                    if sr == 16000:
                        features = features[:, :-1]
                else:
                    features = None

                run = False

            except Exception as ex:

                print('ex', ex, notes, 'Retry')
                # print(librosa.midi_to_note(notes))
                import time
                time.sleep(1)

        return features

# class FishmanGuitarDatabase():

#     def __init__ (self, path='D:/guitar2midi/data/fishman/single_notes/single_notes.pkl', preload_audio=False):


#         import pandas as pd

#         self.df = pd.read_pickle(path)        
#         self.chords = pd.read_json('chord_classes.json')
#         self.chords['poly'] = [len(x) for x in self.chords.midi_notes]
#         self.preload_audio = preload_audio

#         def generate_new_path(old_path, path):
#             #print(old_path)
#             new_root = '/'.join(path.split('/')[:-1])
#             filename = '/'.join(old_path.split('/')[-2:])
#             new_path = '/'.join((new_root, filename))
#             return new_path

#         self.df.index = [generate_new_path(x, path) for x in self.df.index]
#         if self.preload_audio:

#             self.audio_dict = {}

#             for file in self.df.index:
#                 self.audio_dict[file] = librosa.load(file, sr=44100)[0]


#     def sample_chord(self,
#                      notes=None,
#                      samples=8192,
#                      max_delay=4096,
#                      min_delay=0,
#                      pre_roll=64, #samples
#                      delay=None, # delay == attack
#                      sr=44100,
#                      poly='random',
#                      split='train',
#                      padding='noise',
#                      crop_end=0,
#                      crop_end_min=0,
#                      crop_end_max=0,
#                      random_roll=False,
#                      noise_prob=0,
#                      #compensate_offset=True, # load notes with the exact starting point that the midi pickup detected #TODEL obsolete with pre_roll param
#                      #res_type=None, 
#                      #noise=True,
#                      #arpeggio_prob=0,#
#                      #random_crop=False,
#                      #return_audio=False,
#                      extract_features=False,
#                      verbose=True
#                     ):

#         run = True

#         while run == True:

#             try:
#                 import random
#                 #TODELETE
#                 #audio = np.zeros(samples)

#                 if poly == 'random':
#                     poly = np.random.randint(1, 7)
#                     if verbose:
#                         print(poly)

#                 notes = self.chords[self.chords.poly==poly].sample(1).midi_notes.values[0]
#                 if verbose: print('notes', notes)

#                 def get_note(note,
#                              delay=delay,
#                              min_delay=min_delay,
#                              max_delay=max_delay,
#                              crop_end=crop_end,
#                              crop_end_max=crop_end_max,
#                              crop_end_min=crop_end_min
#                             ):

#                     single_note_df = self.df[(self.df.note==note) & (self.df.split==split)].sample(1)
#                     single_note_file = single_note_df.index[0]
#                     offset = int((single_note_df.pre_roll_ms.values[0] / 1000) * sr) # convert to samples

#                     if verbose: print('file', single_note_file, 'offset', offset)
#                     if self.preload_audio:
#                         single_note_audio = self.audio_dict[single_note_file][:samples+sr] # +sr (== +1 below) is just a hack for now when we work with
#                                                                                                             # very short frames anyway, has to be improved,
#                                                                                                             # if/when we work with longer sequences
#                     else:
#                         single_note_audio = librosa.load(single_note_file, sr=sr, duration=(samples/sr)+1)[0] # +1 is just a hack for now when we work with
#                                                                                                             # very short frames anyway, has to be improved,
#                                                                                                             # if/when we work with longer sequences
#                     # attack is at postion single_note_audio[offset] now


#                     single_note_audio = np.pad(single_note_audio, (sr, 0), mode='constant') # +1 is just a hack for now when we work with
#                                                                                                             # very short frames anyway, has to be improved,
#                                                                                                             # if/when we work with longer sequences

#                     #attack is at postion single_note_audio[sr + offset] now

#                     single_note_audio = np.pad(single_note_audio, (0, sr), mode='constant') # +1 is just a hack for now when we work with
#                                                                                                             # very short frames anyway, has to be improved,
#                                                                                                             # if/when we work with longer sequences

#                         #assert single_note_audio.shape[0] == samples


#                     # data augmentation 
# #                     fx = (
# #                         AudioEffectsChain()
# #                         .bandpass(frequency=np.random.randint(100, 1000), q=np.random.randint(1,3))
# #                     )

# #                     single_note_audio = fx(single_note_audio)
# #                     single_note_audio *= np.random.uniform(0.5, 1)

#                     if delay == 'random' or None:
#                         if max_delay>0:
#                             delay = np.random.randint(min_delay, max_delay)
#                         else:
#                             delay = 0
#                     else:
#                         delay = delay

#                     if verbose: print('delay', delay)

#                     out = np.zeros(samples)
#                     out[delay:] = single_note_audio[sr+offset:sr+offset+samples-delay] # alles von attack bis ende
#                     out[:delay] = single_note_audio[sr+offset-delay:sr+offset] # alles was reinpasst von anfang bis attack

#                     if padding == 'noise':
#                         #alles vor pre_roll zu noise...
#                         #single_note_audio[:attack-pre_roll] = np.random.randn(single_note_audio[:attack-pre_roll].shape[0])*np.random.rand(1)*1e-3
#                         out[:delay-pre_roll] = np.random.randn(out[:delay-pre_roll].shape[0])*np.random.rand(1)*1e-3
#                     else:
#                         #...oder zero
#                         #single_note_audio[:attack-pre_roll] = 0
#                         out[:delay-pre_roll] = 0

#                     if crop_end != 'off':
#                         if crop_end=='random':
#                             crop_end = np.random.randint(crop_end_min, crop_end_max)
#                         if padding == 'noise':
#                             #alles vor pre_roll zu noise...
#                             out[-crop_end:] = np.random.randn(out[-crop_end:].shape[0])*np.random.rand(1)*1e-3
#                         else:
#                             #...oder zero
#                             out[-crop_end:] = 0


#                     return out


#                 audio = np.sum([get_note(note) for note in notes], axis=0)/poly

#                 if extract_features:
#                     features =  abs(librosa.cqt(
#                         audio,
#                         sr,
#                         n_bins=84*10,
#                         bins_per_octave=12*10,
#                         hop_length=64,                        
#                     ))
#                 else:
#                     features = None

#                 run = False

#             except Exception as ex:

#                 print('ex', ex, notes, 'Retry')
#                 #print(librosa.midi_to_note(notes))
#                 import time
#                 time.sleep(1)

#         # TODO renam this to overall delay
#         if random_roll:
#             shift = np.random.randint(0, min_delay-pre_roll)
#             audio = audio[shift:]
#             audio = np.pad(audio, (0, samples-audio.shape[0]), mode='constant')


#         if verbose: print(notes)

#         if np.random.rand(1) <= noise_prob:
#             print('Noise!')
#             audio = np.random.randn(samples)*np.random.rand(1)
#             notes = []

#         return audio, features, notes
