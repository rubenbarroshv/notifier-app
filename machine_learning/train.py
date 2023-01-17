
import pickle
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
from torch.utils.data import Dataset
from keras.models import Model
from keras.layers import Input, Dense
"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class config:
    seed=2022
    num_fold = 5
    sample_rate= 32_000
    n_fft=1024
    hop_length=512
    n_mels=64
    duration=7
    num_classes = 2
    train_batch_size = 32
    valid_batch_size = 64
    model_name = 'resnet50'
    epochs = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 1e-4

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dataset_csv():
    labels, assets = [], []
    for root, dir, file in os.walk(r'C:\Marco\Hackathon\dataset\slider'):
        for files in file:
            if root.endswith('abnormal'):
                labels.append(1)
            else:
                labels.append(0)
            assets.append(os.path.join(root, files))
            # print(os.path.split(os.path.split(root)[0])[1])
            # print(file)
    all_df = pd.DataFrame(data={'paths': assets, 'label': labels})
    all_df.to_csv(f"./dataset/all_audios.csv", sep=',', index=False)


class CustomDataset(Dataset):
    def __init__(self, df, transformation, target_sample_rate, duration):
        self.df = df
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * duration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        audio_path = self.df.iloc[index]['paths']

        signal, sr = torchaudio.load(audio_path)  # loaded the audio


        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)


        if signal.shape[0] > 1:
            signal = torch.mean(signal, axis=0, keepdim=True)


        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]

        # If it is less than the required number of samples, we pad the signal
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding)


        mel = self.transformation(signal)

        # For pretrained models, we need 3 channel image, so for that we concatenate the extracted mel
        image = torch.cat([mel, mel, mel])

        # Normalized the image
        max_val = torch.abs(image).max()
        image = image / max_val

        label = torch.tensor(self.df.iloc[index]['label'])

        return image, label



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 54, 64)
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
        x = self.fc2(x)

        return x


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def train(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()

    running_loss = 0
    loop = tqdm(data_loader, position=0)
    for i, (mels, labels) in enumerate(loop):
        mels = mels.to(device)
        labels = labels.to(device)

        outputs = model(mels)
        _, preds = torch.max(outputs, 1)

        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

        loop.set_description(f"Epoch [{epoch + 1}/{config.epochs}]")
        loop.set_postfix(loss=loss.item())

    return running_loss / len(data_loader)


def valid(model, data_loader, device, epoch):
    model.eval()

    running_loss = 0
    pred = []
    label = []

    loop = tqdm(data_loader, position=0)
    for mels, labels in loop:
        mels = mels.to(device)
        labels = labels.to(device)

        outputs = model(mels)
        _, preds = torch.max(outputs, 1)

        loss = loss_fn(outputs, labels)

        running_loss += loss.item()

        pred.extend(preds.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())

        loop.set_description(f"Epoch [{epoch + 1}/{config.epochs}]")
        loop.set_postfix(loss=loss.item())

    valid_f1 = f1_score(label, pred, average='macro')

    return running_loss / len(data_loader), valid_f1





########################################################################
if __name__ == "__main__":

    print(config.device)

    all_df=pd.read_csv(r'C:\Marco\Hackathon\dataset\all_audios.csv')

    df, test_df = train_test_split(all_df,stratify=all_df['label'], test_size=0.1, random_state=config.seed)
    df = df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=config.num_fold)
    for k, (_, val_ind) in enumerate(skf.split(X=df, y=df['label'])):
        df.loc[val_ind, 'fold'] = k

    import torchaudio
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate,
                                                           n_fft=config.n_fft,
                                                           hop_length=config.hop_length,
                                                           n_mels=config.n_mels)

    




    for fold in range(config.num_fold):

        train_df = df[df['fold'] != fold].reset_index(drop=True)
        valid_df = df[df['fold'] == fold].reset_index(drop=True)

        train_dataset = CustomDataset(train_df, mel_spectrogram, config.sample_rate, config.duration)
        valid_dataset = CustomDataset(valid_df, mel_spectrogram, config.sample_rate, config.duration)

        train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False)


        model = Model().to(config.device)

        optimizer = Adam(model.parameters(), lr=config.learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)

        best_valid_f1 = 0
        for epoch in range(config.epochs):
            train_loss = train(model, train_loader, optimizer, scheduler, config.device, epoch)
            valid_loss, valid_f1 = valid(model, valid_loader, config.device, epoch)
            if valid_f1 > best_valid_f1:
                print(f"Validation F1 Improved - {best_valid_f1} ---> {valid_f1}")
                torch.save(model.state_dict(), f'./model/model_{fold}.bin')
                print(f"Saved model checkpoint at ./model/model_{fold}.bin")
                best_valid_f1 = valid_f1
        print("=" * 30)
        print("Training Fold - ", fold)
        print("=" * 30)
        print(f'Best F1 Score: {best_valid_f1:.5f}')

        gc.collect()
        torch.cuda.empty_cache()
