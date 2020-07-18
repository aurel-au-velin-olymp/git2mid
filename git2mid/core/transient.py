import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wavio
import librosa
import tempfile
import glob

def transient_detection(arrays: list, rate=44100):
    '''Takes a list of wav-arrays'''



    temp = tempfile.TemporaryDirectory(dir='')
    temp_dir = '' + temp.name
    
    for i, array in enumerate(arrays):
        wav = wavio.write('{}/{}.wav'.format(temp_dir, str(i)), array, rate, sampwidth=3)    
    
    # process    
    #print(temp_dir)
    command = 'trans\\transient-detection.exe -inDir={} -outDir={}'.format(temp_dir, temp_dir)
    os.system(command)

    files = glob.glob(temp_dir + '/**.wav', recursive=True)
    #print(files)
    #fileName = "FS_Lick7_FN_Lage"
    
    outs = []
    wavs = []
    
    for file in files:
        
        #print(file)

        data = pd.read_csv(file[:-4] + ".csv", header=None, engine='python')
        wav, _ = librosa.load(file[:-4] + '.wav', sr=44100)



        transients = np.where(data[2] == 1)
        out = np.zeros(wav.shape[0])

        for x in transients:
            out[x*64] = 1
        
        outs.append(out)
        wavs.append(wav)
        
    temp.cleanup()
    
    return outs, wavs

def transient_dection_from_files(path='cubase/trans/'):

    files = librosa.util.find_files(path)
    wavs = [librosa.load(file, sr=44100)[0] for file in files[0:7]]
    #trans = [np.load(tran[:-4] + '_trans.npy') for tran in files[0:-1]]
    #wavs = [librosa.effects.trim(wav, top_db=30)[0] for wav in wavs]

    trans, _  = transient_detection(wavs)

    for f, t in zip(files, trans):
        np.save(f[:-4] + '_trans', t)

def sample_chord_transient(transient_db, length=44100, test=True, noise=True):
    
    'takes a list of wavs assuming six strings and noise'
    
    if test:
        wavs = transient_db.wavs_test
        trans = transient_db.trans_test
    else:
        wavs = transient_db.wavs_train
        trans = transient_db.trans_test

    try_it = True
    
    while try_it:
        try:
            poly = np.random.randint(1,7)
            strings = np.random.choice([0,1,2,3,4,5], replace=False, size=poly)
            chord = np.zeros(length)
            label = np.zeros(length)

            for string in strings:
                random_start = np.random.randint(0, wavs[string].shape[0]-length)
                random_end = random_start+length
                chord += wavs[string][random_start:random_end]
                label += trans[string][random_start:random_end]

            # noise
            if noise:
                random_start = np.random.randint(0, wavs[6].shape[0]-length)
                random_end = random_start+length
                chord += wavs[6][random_start:random_end]

            chord /= np.max(abs(chord))

            if not np.any(np.isnan(chord)):
                try_it = False
            else:
                print('Nan found!')

        except Exception as ex:
            print(ex, 'try again.')
            import time
            time.sleep(1)

    return chord, label

def downsample_labels(label):
    
    label_segments = []

    for i in range(0, label.shape[0], 64):
        segment = label[i:i+64]
        if np.max(segment) == 1:
            label_segments.append(1)
        else:
            label_segments.append(0)
            
    return np.stack(label_segments)

class TransientDB:
    
    '''assumes a path with files that are
    a long wav file for every string named 1.wav, 2.wav, etc. and
    a 1_trans.npy, etc. for every wav file '''
        
    def __init__ (self, path='cubase/trans/', sr=44100, load_db=True, padding_mode='wrap', test_split=0.5):
        
        self.path = path
        self.sr = sr
        self.files = librosa.util.find_files(self.path)
        self.wavs = [librosa.load(file, sr=self.sr)[0] for file in self.files]
        
        if load_db:
            self.trans = [np.load(tran[:-4] + '_trans.npy') for tran in self.files]
        else:
            transient_dection_from_files()
        
        assert [wav.shape[0] for wav in self.wavs] == [trans.shape[0] for trans in self.trans]        

        trim_indices = [librosa.effects.trim(wav)[1] for wav in self.wavs]
        self.wavs = [wav[index[0]:index[1]] for index, wav in zip(trim_indices, self.wavs)]
        self.trans = [trans[index[0]:index[1]] for index, trans in zip(trim_indices, self.trans)]

        lens = [wav.shape[0] for wav in self.wavs]

        split_indices = [int(wav.shape[0]*test_split) for wav in self.wavs]

        self.wavs_train = [wav[:index] for index, wav in zip(split_indices, self.wavs)]
        self.trans_train = [trans[:index] for index, trans in zip(split_indices, self.trans)]
        self.wavs_test = [wav[index:] for index, wav in zip(split_indices, self.wavs)]
        self.trans_test = [trans[index:] for index, trans in zip(split_indices, self.trans)]

        self.lens_train = [wav.shape[0] for wav in self.wavs_train]
        self.lens_test = [wav.shape[0] for wav in self.wavs_test]

        max_len_train = max(self.lens_train)
        max_len_test = max(self.lens_test)

        self.wavs_train = [np.pad(wav, (0, max_len_train-wav.shape[0]), mode=padding_mode) for wav in self.wavs_train]
        self.trans_train = [np.pad(trans, (0, max_len_train-trans.shape[0]), mode=padding_mode) for trans in self.trans_train]
        self.wavs_test = [np.pad(wav, (0, max_len_test-wav.shape[0]), mode=padding_mode) for wav in self.wavs_test]
        self.trans_test = [np.pad(trans, (0, max_len_test-trans.shape[0]), mode=padding_mode) for trans in self.trans_test]
        
        
        self.lens_train_seconds = [(wav.shape[0] * (1-test_split)) / self.sr for wav in self.wavs]
        self.lens_test_seconds = [(wav.shape[0] * test_split) / self.sr for wav in self.wavs]