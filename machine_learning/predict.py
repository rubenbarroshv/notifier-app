
import os
import gc
import sys
import glob
import numpy as np
import librosa
import librosa.core
from sklearn.model_selection import train_test_split, StratifiedKFold

import librosa.feature
import yaml
import logging
import random
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from keras.models import Model
from keras.layers import Input, Dense
"""
Standard output is logged in "baseline.log".
"""

class config:
    sample_rate= 32_000
    n_fft=1024
    hop_length=512
    n_mels=64
    duration=7
    num_classes = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'




if __name__ == "__main__":

    print(config.device)


    import torchaudio
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate,
                                                           n_fft=config.n_fft,
                                                           hop_length=config.hop_length,
                                                           n_mels=config.n_mels)


    num_samples = config.sample_rate * config.duration
    audio_path= pd.read_csv(r'C:\Marco\Hackathon\dataset\all_audios.csv')['paths'][569]
    signal, sr = torchaudio.load(audio_path)  # loaded the audio

    if sr != config.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
        signal = resampler(signal)


    if signal.shape[0] > 1:
        signal = torch.mean(signal, axis=0, keepdim=True)


    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]

    # If it is less than the required number of samples, we pad the signal
    if signal.shape[1] < num_samples:
        num_missing_samples = num_samples - signal.shape[1]
        last_dim_padding = (0, num_missing_samples)
        signal = F.pad(signal, last_dim_padding)


    mel = mel_spectrogram(signal)

    # For pretrained models, we need 3 channel image, so for that we concatenate the extracted mel
    image = torch.cat([mel, mel, mel])

    # Normalized the image
    max_val = torch.abs(image).max()
    image = image / max_val

    np.savetxt('sound_file5.txt', image.flatten())
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(128*8*54, 64)
            self.fc2 = nn.Linear(64, config.num_classes)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))

            return x


    model = Model()
    model.load_state_dict(torch.load(f'./model/model_1.bin'))
    model.to(config.device)
    model.eval()
    with torch.no_grad():
        image = image.to(config.device)
        image = torch.unsqueeze(image, 0)

        predicted = model(image)
        result = torch.max(predicted, 1)

