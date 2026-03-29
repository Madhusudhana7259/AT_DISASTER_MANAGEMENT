import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )

        self.fc = nn.Linear(128 * 8 * 8, feature_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNLSTM(nn.Module):
    def __init__(self, feature_dim=128, hidden_dim=128, num_layers=1):
        super().__init__()

        self.encoder = CNNEncoder(feature_dim=feature_dim)

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # reshape for CNN
        x = x.view(B * T, C, H, W)

        features = self.encoder(x)  # (B*T, feature_dim)

        # reshape for LSTM
        features = features.view(B, T, -1)

        lstm_out, _ = self.lstm(features)

        # take last time step
        last_out = lstm_out[:, -1, :]

        out = self.classifier(last_out)

        return out