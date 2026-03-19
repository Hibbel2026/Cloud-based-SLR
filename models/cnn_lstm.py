import torch
import torch.nn as nn
from torchvision import models


class CNN_LSTM(nn.Module):

    def __init__(self, num_classes):

        super(CNN_LSTM, self).__init__()

        # # ===== CNN BACKBONE =====
        # resnet = models.resnet50(pretrained=True)

        # # remove classification layer
        # self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # resnet = models.resnet50(pretrained=True)
        
        resnet = models.resnet50(pretrained=True)


        # ÄNDRA INPUT TILL 6 KANALER
        # resnet.conv1 = nn.Conv2d(
        #     6, 64, kernel_size=7, stride=2, padding=3, bias=False
        # )
        
        old_weights = resnet.conv1.weight.data.clone()

        resnet.conv1 = nn.Conv2d(
            6, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            # RGB = pretrained
            resnet.conv1.weight[:, :3, :, :] = old_weights
            
            # Skeleton = random small init
            resnet.conv1.weight[:, 3:, :, :] = old_weights.mean(dim=1, keepdim=True).repeat(1,3,1,1) * 0.1

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
            
        # ===== FREEZE CNN =====

        # Freeze early layers
        for param in list(self.cnn.parameters())[:-30]:
            param.requires_grad = False

        # 🔥 UNFREEZE sista layers (VIKTIG)
        for param in list(self.cnn.parameters())[-30:]:
            param.requires_grad = True



        self.feature_dim = 2048
        self.feature_norm = nn.BatchNorm1d(self.feature_dim)

        # # ===== LSTM =====
        # self.lstm = nn.LSTM(
        #     input_size=self.feature_dim,
        #     hidden_size=256,
        #     num_layers=2,
        #     batch_first=True,
        #     dropout=0.3
        # )
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True   #  NY
        )


        # ===== DROPOUT =====
        self.dropout = nn.Dropout(0.5)

        # ===== CLASSIFIER =====
        # self.fc = nn.Linear(256, num_classes)

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),  # 🔥 *2 pga bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):

        # x shape
        # (batch, seq_len, C, H, W)

        batch_size, seq_len, C, H, W = x.size()

        # merge batch and seq
        x = x.reshape(batch_size * seq_len, C, H, W)

    

        # CNN features
        features = self.cnn(x)

        # 
        features = features.squeeze(-1).squeeze(-1)

        # reshape
        features = features.view(batch_size, seq_len, self.feature_dim)

        # flatten för BatchNorm
        features = features.view(-1, self.feature_dim)

        # normalize
        features = self.feature_norm(features)

        # reshape tillbaka
        features = features.view(batch_size, seq_len, self.feature_dim)
                
        # LSTM
        lstm_out, _ = self.lstm(features)

        # last time step
        out = torch.mean(lstm_out, dim=1)
        
        # APPLY DROPOUT
        out = self.dropout(out)

        # classifier
        out = self.fc(out)

        return out