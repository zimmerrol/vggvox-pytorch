# Copyright (c) 2018 Roland Zimmermann
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from scipy import signal
import numpy as np
from scipy.io import wavfile

def load_audio_sample(file_name, sampling_rate=16000):
    rate, data = wavfile.read(file_name)
    if len(data.shape) == 2:
        data = data.mean(1)
    if rate != sampling_rate:
        data = signal.resample(data, int(len(data) / rate * sampling_rate))
    return data

def create_audio_sample(data, sampling_rate=16000, sample_duration=4, sample_positon='random'):
    if sample_positon == 'random':
        if len(data) > sample_duration*sampling_rate:
            start = np.random.randint(0, len(data) - sample_duration*sampling_rate)
        else:
            start = 0
    elif sample_positon == 'beginning':
        start = 0
    elif sample_positon == 'end':
        start = max(0, len(data) - sample_duration*sampling_rate)
    else:
        raise ValueError('Unknown value for sample_positon. Allowed values are: random, beginning, end')
        
    end = start + sample_duration*sampling_rate    
    data = data[start:end]

    # zero padding for shorter samples
    if len(data) < sample_duration*sampling_rate:
        zeros =  np.zeros(sample_duration*sampling_rate-len(data))
        data = np.concatenate((data, zeros))
    
    return data

def extract_spectrum(data, sampling_rate=16000, sample_duration=4):   
    f, t, data = signal.stft(data, fs=sampling_rate, window=signal.hamming(int(25e-3*sampling_rate)), nfft=510, nperseg=int(25e-3*sampling_rate), noverlap=int(15e-3*sampling_rate))
    
    data = data[:, :sample_duration*100]
    
    return data

def extract_spectrum_from_file(file_name, sampling_rate=16000, sample_duration=4, sample_positon='random'):
    data = load_audio_sample(file_name, sampling_rate)
    create_audio_sample(data, sampling_rate, sample_duration, sample_positon)
    return extract_spectrum(data)