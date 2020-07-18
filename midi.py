import mido
import pandas as pd
import numpy as np


def midi_file_to_one_many_hot(midi, wav, sr=44100, n_notes=37, lowest_note=40):
    
    import pandas as pd
    
    """Takes midi file and returns many hot labels in full resultion"""
    """Timing etc stimmt, in Cubase gerüft"""
    from mido import tick2second
    from mido import Message, MidiFile, MidiTrack


    mid = MidiFile(midi)
    midi_df = pd.DataFrame(columns=['type', 'note', 'velocity', 'time_delta_seconds'])
    time_offset = 0
    
    for msg in mid:
        if msg.is_meta==True and msg.type=='set_tempo':
            tempo = msg.tempo
        # stupid jam origin does not output ANY 'note_off' uses note_on with vel==0
        elif msg.type == 'note_on' or msg.type == 'note_off':
            if msg.type == 'note_on' and msg.velocity != 0:
                midi_df.loc[len(midi_df)] = ['note_on', msg.note, msg.velocity, msg.time+time_offset] 
            else:
                midi_df.loc[len(midi_df)] = ['note_off', msg.note, msg.velocity, msg.time+time_offset] 
            time_offset = 0
        # wenn zwischen zwei noten zeitinformation kommt (pitchbend etc.)
        elif hasattr(msg, 'time'):
            time_offset += msg.time

    midi_df['time'] = midi_df.time_delta_seconds.cumsum().shift(0).fillna(0)
    midi_df['time_samples_sr'] = np.round(midi_df['time'] * sr).astype('int')
    
    
    if midi_df['type'][0] != 'note_on':
        midi_df = midi_df.iloc[1:]

    piano_roll = np.zeros((n_notes, wav.shape[0]),dtype=np.int8)#, int(midi_df.time_samples_sr.max()+1)])))


    for note in range(0, n_notes):
        note_df = midi_df[midi_df.note==note+lowest_note]
        note_times = note_df.time_samples_sr.values.reshape(-1,2)
        note_vels = note_df.velocity.values.reshape(-1,2)
        for times, vels in zip(note_times, note_vels):
            piano_roll[note, int(times[0]):int(times[1])] = vels[0]
    
    
    on_off = np.zeros((wav.shape[0], 3),dtype=np.int8)
    
    on_off[:,0] = 1

    on = midi_df[midi_df.type=='note_on'].time_samples_sr.values.astype('int')
    off = midi_df[midi_df.type=='note_off'].time_samples_sr.values.astype('int')
    on_off[on, 1] = 1 
    on_off[off, 2] = 1
    
    return piano_roll, on_off, midi_df


def find_onset_in_row(row, threshold=0.5, attack=100, release=10):
    
    """helper for play_midi"""
    
    ons  = []
    offs = []
    
    hold = False
    
    for i, t in enumerate(row):
        
        if hold == False and t > threshold and sum(row[i:i+attack] > threshold) == attack:
            ons.append(i)
            hold = True
            
        if hold == True and t <= threshold and sum(row[i:i+release] <= threshold) == release:
            offs.append(i)
            hold = False
        if hold:
            t = 1
        else:
            t = 0
    return ons, offs


def play_midi(out, attack=1, release=1, threshold=0.5, sr=44100, hop_size=64, outport=None):
    
    """Takes many hot an plays it back"""

    import pandas as pd
    import mido
    
    try:
        outport = mido.open_output('loopMIDI Port 1')
    except:
        outport = outport

    midi_df = pd.DataFrame(columns=['type', 'note', 'time'])


    for i, row in enumerate(out.T):
        ons, offs = find_onset_in_row(row, attack=attack, release=release, threshold=threshold)
        for on in ons:
            midi_df.loc[len(midi_df)] = ['note_on', i+40, on]    
        for off in offs:
            midi_df.loc[len(midi_df)] = ['note_off', i+40, off]

    midi_df = midi_df.sort_values('time')

    timestep_ms = 1000/sr * hop_size
    midi_df['time_ms'] = midi_df['time'] * timestep_ms

    midi_df['time_delta_ms'] = midi_df.time_ms.diff().shift(-1).fillna(0)



        
    return midi_df

def play_midi_df(midi_df, outport):
    for row in midi_df.iterrows():
        msg = mido.Message(row[1].type, note=row[1].note)#, velocity=int(vels[i]* 127))
        import time
        time.sleep(row[1].time_delta_ms / 1000)
        outport.send(msg)
        
def midi_panic(outport):
    for i in range(127):
        #print(int(out[i]* 127))
        msg = mido.Message('note_off', note=i)#, velocity=int(vels[i]* 127))
        outport.send(msg)
        