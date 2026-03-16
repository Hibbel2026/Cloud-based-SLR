import torch
import torch.nn as nn
from torchvision import models


class CNN_LSTM(nn.Module):

    def __init__(self, num_classes):

        super(CNN_LSTM, self).__init__()

        # ===== CNN BACKBONE =====
        resnet = models.resnet50(pretrained=True)

        # remove classification layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        self.feature_dim = 2048


        # ===== LSTM =====
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )


        # ===== CLASSIFIER =====
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):

        # x shape
        # (batch, seq_len, C, H, W)

        batch_size, seq_len, C, H, W = x.size()

        # merge batch and seq
        x = x.view(batch_size * seq_len, C, H, W)

        # CNN features
        with torch.no_grad():
            features = self.cnn(x)

        features = features.view(batch_size, seq_len, self.feature_dim)

        # LSTM
        lstm_out, _ = self.lstm(features)

        # last time step
        out = lstm_out[:, -1, :]

        # classifier
        out = self.fc(out)

        return out