import librosa
import numpy as np
import pandas as pd

sr = 44100

top_chords = pd.read_json('data/chords/top_chords.json')

def samples_to_ms(samples, sr=44100):
    return samples/sr * 1000

def ms_to_samples(ms, sr=44100):
    return ms * sr / 1000


def trim_wav(wav, top_db = 60):
    
    wav, index = librosa.effects.trim(wav, top_db=top_db)
    
    return wav

def normalize_wav(wav):
    
    return wav / np.max(abs(wav))


def noise_gate(wav, frame_length, top_db=30):
    
    wav = librosa.effects.remix(wav, librosa.effects.split(wav, top_db=top_db, frame_length=frame_length, hop_length=10))
    
    return wav


def preload_wav(folder_path, frame_length, sr, sampling = False, sample_size = 1000, top_db=30, crop_start=0.):
    
    crop_start *= sr
    
    file_list = librosa.util.find_files(folder_path)    
    wav_list = [librosa.load(wav, sr = sr)[0][crop_start:] for wav in file_list]
    wav_list = [noise_gate(wav, top_db=top_db, frame_length=frame_length) for wav in wav_list]
    length_list = [len(wav) for wav in wav_list]
    wav_list = [np.pad(wav, pad_width=(0,(max(length_list)-wav.shape[0])), mode='wrap') for wav in wav_list]
    
    return np.array(wav_list)


def preload_wav_and_split(folder_path, frame_length, sr, sampling = False, sample_size = 1000, top_db=30, crop_start=0.):
    
    crop_start *= sr
    
    file_list = librosa.util.find_files(folder_path)    
    wav_list = [librosa.load(wav, sr = sr)[0][crop_start:] for wav in file_list]
    wav_list = [noise_gate(wav, top_db=top_db, frame_length=frame_length) for wav in wav_list]
    wav_list_a = []
    wav_list_b = []
    
    for i in wav_list:
        
        half_length = int(np.floor(i.shape[0]/2))
        wav_a = i[0:half_length]
        wav_b = i[half_length:]
        wav_list_a.append(wav_a)
        wav_list_b.append(wav_b)
    
    length_list_a = [len(wav) for wav in wav_list_a]
    length_list_b = [len(wav) for wav in wav_list_b]

    wav_list_a = [np.pad(wav, pad_width=(0,(max(length_list_a)-wav.shape[0])), mode='wrap') for wav in wav_list_a]
    wav_list_b = [np.pad(wav, pad_width=(0,(max(length_list_b)-wav.shape[0])), mode='wrap' ) for wav in wav_list_b]
    
    
    return np.array(wav_list_a), np.array(wav_list_b)


def get_note_from_filename(file, database='IDMT'):
    
    if database=='IDMT':
        file = file.split('\\')[-1]
        note = int(file.split('-')[1][:2])
    elif database=='own':
        file = file.split('\\')[-1]
        note = int(file[-6:-4]) + 39
        
    return note


def build_database(folder, database='IDMT', top_db=30, sr=44100, split='full', train_split=0.5):
    
    '''
    takes a folder and
    returns a list of (concatenated)
    audio files, where index == note (e.g. 0 = low E)
    '''
    wavs = librosa.util.find_files(folder)    
    wavs_dict = {}
    
    for wav in wavs:
        note = get_note_from_filename(wav, database)
        #print(note)
        wav, _ = librosa.load(wav, sr=sr)
        
        if split == 'train':
            wav = wav[:int(wav.shape[0]*train_split)]
        if split == 'test':
            wav = wav[int(wav.shape[0]*train_split):]
            
        wav /= np.max(abs(wav))
        wav = noise_gate(wav, frame_length=1024)
        if note in wavs_dict:
            #print('New note {}.'.format(note))
            wavs_dict[note] = np.hstack((wavs_dict[note], wav))
        else:
            #print('Append file to {}.'.format(note))
            wavs_dict[note] = wav
            
    return [wavs_dict[key] for key in range(40, 40+37)]
            
    return dbs_all
        

def concatenate_dbs(dbs):
    
    dbs_all = []
    
    for i in range(len(dbs[0])):
        temp = []
        for db in dbs:
            temp.append(db[i])
        dbs_all.append(np.hstack(temp))
    
    return dbs_all


def sample_chords_array(array, window_size):
    
    sample_list = []
    
    random_index = np.random.randint(0, array.shape[1]-window_size-1)
    
    for i in range(0, array.shape[0]):
        
        row = array[i,:]
        sample = row[random_index:random_index+window_size]
        sample_list.append(sample)
        
    sample_list = np.array(sample_list)
    sample_list = np.vstack(sample_list)
    
    return sample_list 


def fft(wav):

    signal = normalize_wav(wav)

    N = wav.shape[0]
    win = np.hamming(N)                                                       
    x = signal[0:N] * win                             # Take a slice and multiply by a window

    sp = np.fft.rfft(x)                               # Calculate real FFT

    mag = np.abs(sp) 
    ref = np.sum(win) / 2                             # Reference : window sum and factor of 2
                                                      # because we are using half of FFT spectrum

    s_dbfs = 20 * np.log10(mag / ref)                 # Convert to dBFS
    norm_spec = librosa.util.normalize(s_dbfs)

    return norm_spec


def to_many_hot_encoding(label):
    
    if label != 'noise':

        from keras.utils import to_categorical

        return np.sum(to_categorical(label, num_classes=37), axis=0)
    else:
        return np.zeros(37)

def get_morlet_freqs_for_guitar(freqs):

    freqs_x = librosa.midi_to_hz(range(40, 40+73))

    freqs_index = []

    for x in freqs_x:

        index = abs(freqs - x).argmin()
        freqs_index.append(index)

    return freqs_index


def wavelet(sample_chord, use_complex=False, sr=4000):
    
    import pywt

    dt = 1/sr
    
    if use_complex:

        w = pywt.ContinuousWavelet('cmor')
        w.bandwidth_frequency=50
        w.center_frequency=1

        frequencies = np.hstack([np.arange(2**i, 2**(i+1), 1/60) for i in range(6)])

        git_freqs = get_morlet_freqs_for_guitar(pywt.scale2frequency(w, frequencies)/dt)
        scales = sorted(list(set(frequencies[git_freqs])))
        scales = scales[::-1]#[0:-16]

        W_complete = [] 

        for i, scale in enumerate(scales): 

            W, freqs = pywt.cwt(sample_chord, scales = scale, wavelet=w, sampling_period=1/sr)
            W_complete.append(W)
            freqs_return.append(freqs)

        W_complete = np.vstack(W_complete)
        W = librosa.util.normalize(np.max(abs(W_complete)**2, axis = 1))
        
    else:
        
            #calculate correct wavelet-scales corresponding to the notes of the guitar
        dt = 1/4000  # 4000 Hz sampling
        frequencies = pywt.scale2frequency('morl', np.arange(1,100,0.1)) / dt
        get_morlet_freqs_for_guitar(frequencies)

        # this selects those scales that are closest to the correct frequencies
        guitar_freqs = np.arange(1,100,0.1)[get_morlet_freqs_for_guitar(frequencies)]

        W, _ = pywt.cwt(sample_chord, scales = guitar_freqs[0:-20], wavelet='morl', sampling_period=1/sr)
        W = np.max(abs(W), axis = 1)
        #W = librosa.util.normalize(W)
        #W *= 0.1

    return W


def get_notes(x):
    
    note = x + 40
    
    return librosa.midi_to_note(note)


def get_random_chord(db, poly, window, mode='top_chords', single_note_prob=0.05, noise_prob=0.0):
    
    if np.random.rand() > noise_prob:
    
        indices = []

        if mode == 'random':
            random_notes = np.sort(np.random.choice(range(0,37), poly, replace=False)) #unrestrictes random chords
        if mode == 'top_chords':

            if np.random.rand() < single_note_prob:
                random_notes = np.sort(np.random.choice(range(0,37), 1, replace=False)) 
            else:
                random_notes = []
                while len(random_notes) == 0:
                    random_notes = np.sort(np.array(top_chords.sample(weights=top_chords['count']).midi_notes.values[0])-40)
                    random_notes[random_notes>=37] -= 12
                    random_notes = np.unique(random_notes)

        chords_samples = []

        for note in random_notes:
            # get a random slice of audio
            random_index = np.random.randint(0, db[note].shape[0]-window-1) 
            sample = db[note][random_index:random_index+window]

            # apply random polarity flip == minimal form of data augmentation
            if np.random.rand(1)[0]<0.5:
                sample = -sample

            chords_samples.append(sample)
            indices.append(random_index)

        chord = np.vstack(chords_samples)

        chord = np.sum(chord, axis = 0)
        chord = normalize_wav(chord)
        
    else:
        #chord = np.random.randn(window) # white noise
        chord = noise = np.sum([librosa.core.tone(freq, sr=44100, length=window)*np.random.randn() for freq in np.random.randint(40,22050, size=np.random.randint(1,100))],axis=0) # tonal noise

        random_notes = "noise"

    return chord, random_notes


def generate_data(n, db, window, mode='top_chords'):

    audio_list = []
    labels_list = []

    for i in range(0, n):

        poly = np.random.randint(1,7)
        random_chord, random_notes = get_random_chord(poly=poly, db=db, window=window, mode=mode)

        audio_list.append(random_chord)
        labels_list.append(random_notes)
    
    Y = np.array([to_many_hot_encoding(label) for label in labels_list])
    X_audio = np.array(audio_list)

    return X_audio, Y, labels_list


def evaluate_model(model, X, Y, n = 100, print_all=False, batch_size=64):
    
    print(n)
    
    y_hat = model.predict([X[0:n]], batch_size=batch_size)

    score = 0
    poly_error = 0
    noise_error = 0
    noise_hit = 0

    for i in range(0,n):

        test_sample = i
        result = y_hat[test_sample] # auswählen
        print(result)

        label = (np.vstack(np.where(Y[i]==1)))[0] # label holen
        
        print(result, label)

        result = result.argsort()[::-1][0:len(label)] # top_k von result anschauen, wobei k gleich n_poly ist
        result = np.array(sorted(result))

        temp_poly_error = abs(len(np.where(y_hat[i]>0.5)[0])-len(label))
        poly_error += temp_poly_error

        intersection = np.intersect1d(result,label)
        
        if len(result) == 0 or len(label)==0:
            if len(result) == 0 and len(label) == 0:
                noise_hit += 1
            else:
                noise_error +=1
                
        else:
            score += len(intersection)/len(label)
    
        if print_all == True:
            
            print('========= Sample {} =========='.format(i))
            print('')

            print('ground_truth:', librosa.midi_to_note(label+40))
            print('prediction  :', librosa.midi_to_note(result+40))
            print('___________________')

            print('notes match :', len(intersection))
            print('false n/p   :', (len(np.where(y_hat[i]>0.5)[0]))-len(label))
            print('___________________')

            print('Score:', len(intersection)/len(label))

            print('')
            print('')

    print('score:', score/n, 'poly error:', poly_error/n, 'noise_error', noise_error, 'noise_hit', noise_hit)
    
    return score/n, poly_error/n

def freq_and_amp_to_spec(frequency, amplitude, sr=44100, n_fft=4096*2):


    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    diffs = np.array([frequency-f for f in fft_freqs])

    closest_bins = np.argsort(abs(diffs))[0:2]
    closest_freqs = fft_freqs[closest_bins]

    diff = abs(closest_bins[0]-closest_bins[1])

    weight_bin_1 = abs(closest_bins[0]-frequency)/diff
    weight_bin_2 = abs(closest_bins[1]-frequency)/diff

    out = np.zeros(int(n_fft/2)+1,)
    out[int(closest_bins[0])] = amplitude*weight_bin_1
    out[int(closest_bins[1])] = amplitude*weight_bin_2
    
    return out


def sine_wave(frequency, amplitude, length=8192, Fs=44100):
    #print(frequency, amplitude)
    x = np.arange(length)
    y = np.sin(2 * np.pi * frequency * x / Fs) * amplitude
    
    return y


def fft_librosa(wav):
    
    spec = librosa.stft(wav,win_length=8192, center=False, n_fft=8192, window='hann')
    librosa.fft_frequencies(sr=44100, n_fft=8192)
    
    return np.abs(spec).ravel()

def custom_hz_bins():
    
    note_start = 24
    octs = 11

    Hz = [librosa.midi_to_hz(note_start)]

    for i in range(100*12*octs):
        
        f = Hz[-1] * 1.0005777895065548592967925757932
        
        if f <= 22050:
            Hz.append(f)
        
    return np.array(Hz)


def sinusoidal_extraction_batch(wav_array, maxTracks=150, return_mean=True, reconstruction='direct', digitize=False):
    
    import soundfile as sf
    import pandas as pd
    import shutil
    import os

    try:
        shutil.rmtree('temp/csv', ignore_errors=True) 
        shutil.rmtree('temp/wav', ignore_errors=True) 
        os.mkdir('temp/csv')
        os.mkdir('temp/wav')
    except:
        pass


    
    csv_list = []
    
    for i, wav in enumerate(wav_array):
        sf.write('temp/wav/temp{}.wav'.format(i), wav, samplerate=sr)
        csv_list.append("temp/csv/temp{}.csv".format(i))

    
    #os.system('sinusoidal-extraction.exe -inDir=temp\wav -outDir=temp\csv -maxTracks=' + str(maxTracks))
    os.system('sinusoidal-extraction-0.3.exe -inDir=temp\wav -outDir=temp\csv -maxTracks=150 -oneFrameMode=1')
    
    def read_csv(csv, return_mean=return_mean):
        
        import pandas as pd

        '''reads csv and returns freqs and amplitude arrays'''

        data = pd.read_csv(csv, header=None, engine='python')
        data = np.array(data.values, dtype=np.float)
        data[np.isnan(data)] = 0

        freq = data[:,::2]
        freq = freq.ravel()*22050

        amp = data[:,1::2].ravel()
        
        if digitize == False:
        
            if reconstruction == 'direct' or 'both':
                spec_d = np.sum([freq_and_amp_to_spec(f,a,n_fft=4096*2) for f,a in zip(freq,amp)], axis = 0)
            if reconstruction == 'audio' or 'both':
                audio_reconstruction = np.sum([sine_wave(f,a) for f,a in zip(freq, amp)], axis=0)        
                spec_a = fft_librosa(audio_reconstruction).ravel()
                
        else:
            
            fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=4096*16)
            fft_freqs = custom_hz_bins()
            fft_freqs = custom_hz_bins()[::10]
            
            
            bins = np.digitize(freq, fft_freqs)
            spec_d = np.zeros(fft_freqs.shape[0])
            for b, a in zip(bins, amp):
                spec_d[b] += a
                
            spec_a = None
                
        return spec_d, spec_a


    X = [read_csv(csv) for csv in csv_list]
    X_direct = [x[0] for x in X]
    X_audio = [x[1] for x in X]
    
    return np.stack(X_direct), np.stack(X_audio)


def generate_note(freq, length = 5, string='E', noise_harmonics_level=False, noise_harmonics_freq=False, Fs=44100):
    
    T = 1/Fs # sampling period

    
    #harm_level = [1, 1, 0.8, 0.7, 0.6, 0.5, 0.4]
    harm_dict = dict(
    E = [10, 13, 11, 16, 12, 9, 8],
    A = [15, 14, 25, 17, 10, 7, 3],
    D = [19, 21, 18, 14, 9, 6, 3],
    G = [18, 14, 14, 14, 9, 6, 4],
    H = [15, 35, 24, 21, 18, 9, 6],
    e = [18, 34, 18, 10, 11, 9, 6],
    )
    
    harm_level = harm_dict[string]
    harm_level /= np.sum(harm_level)


    t = length+1
    N = Fs*t # total points in signal

    
    for i, level in enumerate(harm_level):
        
        if noise_harmonics_level:
            level += np.random.randn(1) * 0.005 # add somie nosie to harmonics levels
        if noise_harmonics_freq:
            harm_freq_noise = [0, 0, 0.01, 0.05, 0.1, 0.1]
            freq_harm = freq #+ np.random.randn(1) * harm_freq_noise[i]
        
        omega = 2*np.pi*(freq*i) # angular frequency for sine waves

        t_vec = np.arange(N)*T # time vector for plotting
        
        random_start = np.random.randint(Fs)
        random_end = random_start + length*Fs
        
        y = np.sin(omega*t_vec)*level
        y = y[random_start:random_end]

        if i == 0:
            note = y
        else:
            note += y
    
    return librosa.util.normalize(note)


def generate_chord(midi_notes):
    
    freqs = librosa.midi_to_hz(midi_notes)

        
    chord = []
    
    for i, string in enumerate(['E', 'A', 'D', 'G', 'H', 'e']):
        
        note = generate_note(freqs[i], length=2, string=string)
        chord.append(note)
        
        
    chord = np.sum(chord, axis=0)
    
    fade = (np.arange(chord.size)/1)[::-1]

    
    return chord*fade, freqs

def get_notes(xml):

    import xmltodict

    with open(xml) as fd:
        doc = xmltodict.parse(fd.read())

    events =  doc['instrumentRecording']['transcription']['event']
    pitches = [int(pitch['pitch']) for pitch in events]
    notes = librosa.midi_to_note(pitches)
    return notes

def predict_and_plot_chord(wav_path, labels, model, suffix='', digitize=False):
    

    
    wav, sr = librosa.load(wav_path, sr=44100)
    wav_array = librosa.util.frame(wav, frame_length=8192, hop_length=1024).T

    #wav_array =  librosa.util.normalize(wav_array, axis=1)

    X_direct, X_audio = sinusoidal_extraction_batch(wav_array, reconstruction='both', digitize=digitize)

    y_hat = model.predict(librosa.util.normalize(X_direct, axis=1))
    #y_hat = model.predict(X_audio)
    
    labels = librosa.note_to_midi(labels) - 40

    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 20))
    

    #plt.imshow(np.round(y_hat.T))
    plt.imshow(y_hat.T, origin='lower')
    plt.yticks(np.arange(0,37), labels=librosa.midi_to_note(np.arange(40, 37+40)))
    
    for label in labels:
        for row in range(y_hat.shape[0]):
            plt.plot(row, label, 'r+')
    
    plt.savefig(wav_path[:-4] + '_' + suffix + '.png')
    plt.show()


def get_note_names_c_major():

    note_names = librosa.midi_to_note(np.arange(40,40+37))
    note_names_c_major =[]

    for note in note_names:
        if not '#' in note:
            note_names_c_major.append(note)
        else:

            note_names_c_major.append(None)

    return note_names_c_major