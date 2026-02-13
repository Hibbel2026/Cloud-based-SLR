import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCNN(nn.Module):
    """1D CNN for sign language recognition from MediaPipe landmarks"""
    
    def __init__(self, input_features=225, num_classes=100, dropout=0.5):
        super(TemporalCNN, self).__init__()
        
        # Temporal convolutional layers
        self.conv1 = nn.Conv1d(input_features, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, frames, features]
        # Conv1d expects: [batch_size, features, frames]
        x = x.transpose(1, 2)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # [batch_size, 512]
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x