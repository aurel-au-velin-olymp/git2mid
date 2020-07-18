import pyaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import time


def record(threshold=3, auto_start=True, duration=3):
    
    import pyaudio
    import wave

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024*4
    RECORD_SECONDS = duration
    WAVE_OUTPUT_FILENAME = "temp.wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    
    if auto_start:
        silence = True
    else:
        silence = False
        
    while silence:
        data = stream.read(CHUNK*4)
        data_float = np.frombuffer(data, dtype='B')
        nrg = np.sum(data_float**2)/CHUNK
        #time.sleep(1)
        if nrg > threshold*100:
            #time.sleep(1)
            silence=False
    
    

    print("recording...")
    frames = [data]

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")


    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    
def get_features(audio, return_mean=True, res_type='polyphase', sr=44100, load_temp_audio=True):

    if load_temp_audio:
        audio = get_audio()

    features =  abs(librosa.cqt(
        audio,
        sr,
        n_bins=84*10,
        bins_per_octave=12*10,
        hop_length=64,
        res_type=res_type                
    ))

    if return_mean:
        features = np.mean(features, axis=1)

    return features

def get_audio(sr=44100):
    return librosa.load('temp.wav', sr=sr)[0]