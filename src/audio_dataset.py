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

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import utility as ut
import os
import torch

class AudioDataset(Dataset):
    """Audio dataset"""

    def __init__(self, csv_file, base_audio_path, stft_transform=None):
        self._base_audio_path = base_audio_path
        self._table = pd.read_csv(csv_file)
        self._audio_data = {}
        self._stft_transform = stft_transform

        # removed samples from *.csv whose *.wav files are not available
        indices_to_remove = []
        for idx in range(len(self._table)):
            wav_name = os.path.join(self._base_audio_path, self._table.wav_name[idx],)
            if not os.path.exists(wav_name):
                indices_to_remove.append(idx)

        self._table = self._table.drop(indices_to_remove)
        self._table = self._table.reset_index()

    def __len__(self):
        return len(self._table)

    def __getitem__(self, idx):
        wav_name = os.path.join(self._base_audio_path, self._table.wav_name[idx])
        if wav_name in self._audio_data:
            wav_data = self._audio_data[wav_name]
        else:
            wav_data = ut.load_audio_sample(wav_name)
            self._audio_data[wav_name] = wav_data

        # create sample
        wav_data = ut.create_audio_sample(wav_data)

        audio_stft = ut.extract_spectrum(wav_data)
        audio_stft = np.vstack((audio_stft.real, audio_stft.imag))

        if self._stft_transform:
            audio_stft = self._stft_transform(audio_stft)

        audio_stft = audio_stft.reshape((1, *audio_stft.shape))

        audio_stft = torch.from_numpy(audio_stft.astype(dtype=np.float32))
        labels = torch.from_numpy(np.array(self._table.target[idx]).astype(dtype=np.float32))

        return (audio_stft, labels)