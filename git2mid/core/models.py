from __future__ import division
from __future__ import print_function

import os
import re
import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf

# LSTM

import torch
import torch.nn as nn
import torch.nn.functional as F

import nnAudio
from nnAudio import Spectrogram


class PyTorch(nn.Module):

    def __init__(self, drop_last=True, roll=False, return_cqt=False, output=None):
        super(PyTorch, self).__init__()

        self.cqt = Spectrogram.CQT2010v2(
            sr=44100,
            hop_length=64,
            n_bins=84*10,
            bins_per_octave=12*10,
            norm=1,
            window='hann',
            pad_mode='constant',
            trainable=False,
        )
        
        self.roll = roll
        self.drop_last = drop_last
        self.return_cqt = return_cqt
        self.output = output

        self.bn0 = nn.BatchNorm1d(840)
        self.conv1 = nn.Conv1d(840, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv2 = nn.Conv1d(1024, 512, 1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 48, 1)
        self.bn4 = nn.BatchNorm1d(48)  
        
        self.LSTM = nn.LSTM(48, 512, batch_first=True, num_layers=2, dropout=0.25)        
        self.conv5 = nn.Conv1d(512, 48, 1)
        
        self.dropout_1 = nn.Dropout(p=0.25)
        self.dropout_2 = nn.Dropout(p=0.25)
        self.dropout_3 = nn.Dropout(p=0.25)
        self.dropout_4 = nn.Dropout(p=0.25)
        
    def forward(self, x):
        y = x
        cqt = self.cqt(y)
        x = torch.relu(self.bn1(self.conv1(cqt)))
        x = self.dropout_1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout_2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.dropout_3(x)
        x = torch.relu(self.bn4(self.conv4(x)))
        pre_lstm = self.dropout_4(x)

        x = pre_lstm.permute(0, 2, 1)
        x = self.LSTM(x)[0]
        post_lstm = x.permute(0, 2, 1)

        x = self.conv5(post_lstm)
        
        pred = torch.sigmoid((x))
        
        if self.drop_last:
            x = x[:,:,:-1]
        else:
            x[:,:,1:]
            
        if self.roll:
            pred = torch.roll(pred, -40, dims=1)
        
        if self.output is None:
            if self.return_cqt:
                return pred, cqt
            else:
                return x
        else:
            
            output_dict = {
                
                'pred' : pred,
                'cqt' : cqt,
                'pre_lstm' : pre_lstm,
                'post_lstm' : post_lstm

            }
            
            
            
            return output_dict[self.output]

class PyTorchv1():

    def __init__(self, weights='pytorch_cqt_12.pt', drop_last=True, roll=False, return_cqt=False, output=None):
        #super(Net, self).__init__()

        self.drop_last = drop_last
        self.roll = roll
        self.return_cqt = return_cqt
        self.output = output
        self.model = PyTorch(drop_last=self.drop_last, roll=self.roll, return_cqt=self.return_cqt, output=self.output)
        self.model.cuda()
        self.model.eval()
        self.model.load_state_dict(torch.load(weights), strict=False)


class VaeEncoder:
        
    def __init__ (self):

        self.encoder = tf.keras.models.load_model('vae_cqt_encoder.h5')
        
            
    def encode(self, array):
        
        '''takes cqt features and embeddes them'''
        
        array /= 35
        return self.encoder.predict(array)
    
    
class Guitar2Midi():
        
    def __init__ (self, sequences=True, crop=False):

        if sequences:
            if crop:
                self.model = tf.keras.models.load_model('seq-cqt-lstm-chord-crop.h5')
            else:
                self.model = tf.keras.models.load_model('seq-cqt-lstm-chord-WORKS.h5')
        else:
            self.model = tf.keras.models.load_model('cqt_retrained.h5')
        
            
    def predict(self, array):
        
        '''takes cqt features and predicts many hot'''
        
        return self.encoder.predict(array)[0]
    
    
class Crepe():
    

    def __init__(self, sr=44100):
        
        """
        Build the CNN model and load the weights

        Parameters
        ----------
        model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
            String specifying the model capacity, which determines the model's
            capacity multiplier to 4 (tiny), 8 (small), 16 (medium), 24 (large),
            or 32 (full). 'full' uses the model size specified in the paper,
            and the others use a reduced number of filters in each convolutional
            layer, resulting in a smaller model that is faster to evaluate at the
            cost of slightly reduced pitch estimation accuracy.

        Returns
        -------
        model : tensorflow.keras.models.Model
            The pre-trained keras model loaded in memory
        """
        from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization
        from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense
        from tensorflow.keras.models import Model
        
        models = {
        'tiny': None,
        'small': None,
        'medium': None,
        'large': None,
        'full': None
        }

        # the model is trained on 16kHz audio
        #model_srate = 16000
        self.sr = sr

        #if models[model_capacity] is None:
        capacity_multiplier = 32 #{
#                 'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
#             }[model_capacity]

        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        x = Input(shape=(1024,), name='input', dtype='float32')
        y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

        for l, f, w, s in zip(layers, filters, widths, strides):
            y = Conv2D(f, (w, 1), strides=s, padding='same',
                       activation='relu', name="conv%d" % l)(y)
            y = BatchNormalization(name="conv%d-BN" % l)(y)
            y = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                          name="conv%d-maxpool" % l)(y)
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)

        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)
        y = Dense(360, activation='sigmoid', name="classifier")(y)

        self.model = Model(inputs=x, outputs=y)

        filename = "model-full.h5"
        self.model.load_weights(filename)
        self.model.compile('adam', 'binary_crossentropy')


    def to_local_average_cents(self, salience, center=None):
        """
        find the weighted average cents near the argmax bin
        """

        mapping = (np.linspace(0, 7180, 360) + 1997.3794084376191)

        if salience.ndim == 1:
            if center is None:
                center = int(np.argmax(salience))
            start = max(0, center - 4)
            end = min(len(salience), center + 5)
            salience = salience[start:end]
            product_sum = np.sum(
                salience * mapping[start:end])
            weight_sum = np.sum(salience)
            return product_sum / weight_sum
        if salience.ndim == 2:
            return np.array([self.to_local_average_cents(salience[i, :]) for i in
                             range(salience.shape[0])])

        raise Exception("label should be either 1d or 2d ndarray")


    def to_viterbi_cents(self, salience):
        """
        Find the Viterbi path using a transition prior that induces pitch
        continuity.
        """
        from hmmlearn import hmm

        # uniform prior on the starting pitch
        starting = np.ones(360) / 360

        # transition probabilities inducing continuous pitch
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / np.sum(transition, axis=1)[:, None]

        # emission probability = fixed probability for self, evenly distribute the
        # others
        self_emission = 0.1
        emission = (np.eye(360) * self_emission + np.ones(shape=(360, 360)) *
                    ((1 - self_emission) / 360))

        # fix the model parameters because we are not optimizing the model
        model = hmm.MultinomialHMM(360, starting, transition)
        model.startprob_, model.transmat_, model.emissionprob_ = \
            starting, transition, emission

        # find the Viterbi path
        observations = np.argmax(salience, axis=1)
        path = model.predict(observations.reshape(-1, 1), [len(observations)])

        return np.array([self.to_local_average_cents(salience[i, :], path[i]) for i in
                         range(len(observations))])


    def get_activation(self, audio, center=True, step_size=10,
                       verbose=1):
        """

        Parameters
        ----------
        audio : np.ndarray [shape=(N,) or (N, C)]
            The audio samples. Multichannel audio will be downmixed.
        sr : int
            Sample rate of the audio samples. The audio will be resampled if
            the sample rate is not 16 kHz, which is expected by the model.
        model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
            String specifying the model capacity; see the docstring of
            :func:`~crepe.core.build_and_load_model`
        center : boolean
            - If `True` (default), the signal `audio` is padded so that frame
              `D[:, t]` is centered at `audio[t * hop_length]`.
            - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
        step_size : int
            The step size in milliseconds for running pitch estimation.
        verbose : int
            Set the keras verbosity mode: 1 (default) will print out a progress bar
            during prediction, 0 will suppress all non-error printouts.

        Returns
        -------
        activation : np.ndarray [shape=(T, 360)]
            The raw activation matrix
        """
        model = self.model
        sr = self.sr

        if len(audio.shape) == 2:
            audio = audio.mean(1)  # make mono
        audio = audio.astype(np.float32)
        if sr != 16000:
            # resample audio if necessary
            from resampy import resample
            audio = resample(audio, sr, 16000)

        # pad so that frames are centered around their timestamps (i.e. first frame
        # is zero centered).
        if center:
            audio = np.pad(audio, 512, mode='constant', constant_values=0)

        # make 1024-sample frames of the audio with hop length of 10 milliseconds
        hop_length = int(16000 * step_size / 1000)
        n_frames = 1 + int((len(audio) - 1024) / hop_length)
        frames = as_strided(audio, shape=(1024, n_frames),
                            strides=(audio.itemsize, hop_length * audio.itemsize))
        frames = frames.transpose().copy()

        # normalize each frame -- this is expected by the model
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.std(frames, axis=1)[:, np.newaxis]

        # run prediction and convert the frequency bin weights to Hz
        return model.predict(frames, verbose=verbose)


    def predict(self, audio,
                viterbi=False, center=True, step_size=10):
        """
        Perform pitch estimation on given audio

        Parameters
        ----------
        audio : np.ndarray [shape=(N,) or (N, C)]
            The audio samples. Multichannel audio will be downmixed.
        sr : int
            Sample rate of the audio samples. The audio will be resampled if
            the sample rate is not 16 kHz, which is expected by the model.
        model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
            String specifying the model capacity; see the docstring of
            :func:`~crepe.core.build_and_load_model`
        viterbi : bool
            Apply viterbi smoothing to the estimated pitch curve. False by default.
        center : boolean
            - If `True` (default), the signal `audio` is padded so that frame
              `D[:, t]` is centered at `audio[t * hop_length]`.
            - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
        step_size : int
            The step size in milliseconds for running pitch estimation.
        verbose : int
            Set the keras verbosity mode: 1 (default) will print out a progress bar
            during prediction, 0 will suppress all non-error printouts.

        Returns
        -------
        A 4-tuple consisting of:

            time: np.ndarray [shape=(T,)]
                The timestamps on which the pitch was estimated
            frequency: np.ndarray [shape=(T,)]
                The predicted pitch values in Hz
            confidence: np.ndarray [shape=(T,)]
                The confidence of voice activity, between 0 and 1
            activation: np.ndarray [shape=(T, 360)]
                The raw activation matrix
        """
        activation = self.get_activation(audio,
                                    center=center, step_size=step_size,
                                    verbose=0)
        confidence = activation.max(axis=1)

        if viterbi:
            cents = self.to_viterbi_cents(activation)
        else:
            cents = self.to_local_average_cents(activation)

        frequency = 10 * 2 ** (cents / 1200)
        frequency[np.isnan(frequency)] = 0

        time = np.arange(confidence.shape[0]) * step_size / 1000.0

        return np.mean(frequency)


    