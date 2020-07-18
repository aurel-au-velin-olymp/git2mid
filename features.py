
import soundfile as sf
import pandas as pd
import shutil
import os
import numpy as np
import librosa



def custom_hz_bins():
    
    note_start = 24
    octs = 11

    Hz = [librosa.midi_to_hz(note_start)]

    for i in range(100*12*octs):
        
        f = Hz[-1] * 1.0005777895065548592967925757932
        
        if f <= 22050:
            Hz.append(f)
        
    return np.array(Hz)


def sinusoidal_extraction_batch(wav_array, data_augmentation=False, denoising_threshold=0., pitch_shift_range=1, version='0.4'):
    
    import uuid
    uid = str(uuid.uuid4())

    print(os.getcwd())

    try:
        shutil.rmtree('C:/Users/opauly/prv/guitar2midi/temp/csv', ignore_errors=True) 
        shutil.rmtree('C:/Users/opauly/prv/guitar2midi/temp/wav', ignore_errors=True) 
    except:
        pass
    
    try:
        os.mkdir('C:/Users/opauly/prv/guitar2midi/temp/csv')
        os.mkdir('C:/Users/opauly/prv/guitar2midi/temp/wav')
    except:
        pass
    
    sf.write('C:/Users/opauly/prv/guitar2midi/temp/wav/temp_{}.wav'.format(uid), wav_array.reshape(-1,), samplerate=44100)
    
    #os.system('sinusoidal-extraction.exe -inDir=temp\wav -outDir=temp\csv -maxTracks=' + str(maxTracks))
    os.system('C:/Users/opauly/prv/guitar2midi/sinusoidal-extraction-0.4.exe -inDir=C:/Users/opauly/prv/guitar2midi/temp/wav -outDir=C:/Users/opauly/prv/guitar2midi/temp/csv -maxTracks=150 -oneFrameMode=1'.format(version))
    
    df = pd.read_csv('C:/Users/opauly/prv/guitar2midi/temp/csv/temp_{}.csv'.format(uid), header=None, engine='python')
    #print(len(df))
    data = np.array(df.values, dtype=np.float)
    data[np.isnan(data)] = 0
    #print(data.shape)

    X = []

    fft_freqs = custom_hz_bins()[::10]

    for sample in data:

        freq = sample[::2]
        freq = freq.ravel()*22050

        amp = sample[1::2].ravel()
        bins = np.digitize(freq, fft_freqs)
        spec = np.zeros(fft_freqs.shape[0])
        for b, a in zip(bins, amp):

            
            if a < denoising_threshold:
                a = 0
                
                
            if data_augmentation:

                bin_noise = 0 #np.random.choice(np.arange(-1,1))
                #bin_mult = 
                amp_noise = abs(np.random.normal(1, 0.05))

                b_noisy = b #+ bin_noise
                
                a_noisy = a * amp_noise
                if a_noisy < 0:
                    a_noisy = 0
                    

                try:
                    spec[b_noisy] += a_noisy
                except:
                    spec[b] += a
            
            else:
                
                spec[b] += a
        
        if data_augmentation:
            # pitch shift
            pitch_shift = np.random.choice(np.arange(-pitch_shift_range,pitch_shift_range))
            spec = np.roll(spec, pitch_shift)
            if pitch_shift > 0:
                spec[:pitch_shift] = 0
            if pitch_shift < 0:
                spec[pitch_shift:] = 0
                
            amp_bias = np.random.normal(0, 0.01)
            spec += amp_bias

            

        X.append(spec)

    X = np.vstack(X)      
    
    return X#, data, df


def sinusoidal_extraction_batch_seq(wav_array, data_augmentation=False, denoising_threshold=0., pitch_shift_range=1, version='0.4'):
    
    import soundfile as sf
    import pandas as pd
    import shutil
    import os
    import numpy as np
    import librosa

    import uuid
    uid = str(uuid.uuid4())
    
#     try:
#         shutil.rmtree('C:/Users/opauly/prv/guitar2midi/temp/csv', ignore_errors=True) 
#         shutil.rmtree('C:/Users/opauly/prv/guitar2midi/temp/wav', ignore_errors=True) 
#     except:
#         pass
    
    try:
        os.mkdir('C:/Users/opauly/prv/guitar2midi/temp/csv')
        os.mkdir('C:/Users/opauly/prv/guitar2midi/temp/wav')
    except:
        pass
    
    
    sf.write('C:/Users/opauly/prv/guitar2midi/temp/wav/temp_{}.wav'.format(uid), wav_array.reshape(-1,), samplerate=44100)

    

    
    #os.system('sinusoidal-extraction.exe -inDir=temp\wav -outDir=temp\csv -maxTracks=' + str(maxTracks))
    os.system('C:/Users/opauly/prv/guitar2midi/sinusoidal-extraction-0.4.exe -inDir=C:/Users/opauly/prv/guitar2midi/temp/wav -outDir=C:/Users/opauly/prv/guitar2midi/temp/csv -maxTracks=150 -oneFrameMode=0'.format(version))
    
    df = pd.read_csv('C:/Users/opauly/prv/guitar2midi/temp/csv/temp_{}.csv'.format(uid), header=None, engine='python')
    
        
    os.remove('C:/Users/opauly/prv/guitar2midi/temp/csv/temp_{}.csv'.format(uid)) 
    os.remove('C:/Users/opauly/prv/guitar2midi/temp/wav/temp_{}.wav'.format(uid)) 
    
    
    
    #print(len(df))
    data = np.array(df.values, dtype=np.float)
    data[np.isnan(data)] = 0
    #print(data.shape)

    X = []

    fft_freqs = custom_hz_bins()[::10]

    for sample in data:

        freq = sample[::2]
        freq = freq.ravel()*22050

        amp = sample[1::2].ravel()
        bins = np.digitize(freq, fft_freqs)
        spec = np.zeros(fft_freqs.shape[0])
        for b, a in zip(bins, amp):

            
            if a < denoising_threshold:
                a = 0
                
                
            if data_augmentation:

                bin_noise = 0 #np.random.choice(np.arange(-1,1))
                #bin_mult = 
                amp_noise = abs(np.random.normal(1, 0.05))

                b_noisy = b #+ bin_noise
                
                a_noisy = a * amp_noise
                if a_noisy < 0:
                    a_noisy = 0
                    

                try:
                    spec[b_noisy] += a_noisy
                except:
                    spec[b] += a
            
            else:
                
                spec[b] += a
        
        if data_augmentation:
            # pitch shift
            pitch_shift = np.random.choice(np.arange(-pitch_shift_range,pitch_shift_range))
            spec = np.roll(spec, pitch_shift)
            if pitch_shift > 0:
                spec[:pitch_shift] = 0
            if pitch_shift < 0:
                spec[pitch_shift:] = 0
                
            amp_bias = np.random.normal(0, 0.01)
            spec += amp_bias

            

        X.append(spec)
        #print(spec.shape)

    X = np.vstack(X)      
    
    return X, data, df


def multi_fft_tf(audio_array, fft_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]):
    
    import tensorflow as tf
    
    #audio_array = tf.convert_to_tensor(audio_array, dtype=)

    #with tf.device('/device:GPU:0'):
    fft = [tf.abs(tf.signal.stft(audio_array, frame_length=size, frame_step=64, pad_end=True)) for size in fft_sizes]
    
    return tf.concat(fft, axis=-1)

