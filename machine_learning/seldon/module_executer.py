'''
Custom Module Loader
'''

import mlflow.pyfunc
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import logging

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
        x = torch.sigmoid(self.fc2(x))

        return x

class Executor(mlflow.pyfunc.PythonModel):
    '''
    Yolo model executor
    '''

    def __init__(self, model=None):
        self.model = model


    def predict(self, audio):
        logging.info(f'{np.array(audio).shape}')
        audio=torch.tensor(audio).reshape(torch.Size([3, 64, 438]))
        
        audio = torch.unsqueeze(audio, 0).to(device=config.device, dtype=torch.float)
        # logging.info(f'model: {model}')

        with torch.no_grad():
            predicted = self.model(audio)
            result = torch.max(predicted, 1).indices.item()
            result='defect' if result ==1 else 'normal'
            
            jsonData = {"data": {"class": result, "score": torch.max(predicted, 1).values.item() }}
        return jsonData
    
def _load_pyfunc(model_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model()
    model.load_state_dict(torch.load(model_file))
    model.to(config.device)
    model.eval()
    return Executor(model)
