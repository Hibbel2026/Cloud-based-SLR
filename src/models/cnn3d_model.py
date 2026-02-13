"""
3D CNN Model for Sign Language Recognition
Processes raw video frames directly without MediaPipe preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D Convolutional block with batch norm and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Temporal3DCNN(nn.Module):
    """
    3D CNN for Sign Language Recognition
    
    Input: [batch, channels, frames, height, width]
    Example: [32, 3, 16, 224, 224] (16 frames, 224x224 RGB video)
    
    Architecture:
    - 4 Conv3D blocks with increasing channels (3 -> 64 -> 128 -> 256 -> 512)
    - 3D Max pooling after each block
    - Adaptive average pooling for variable input sizes
    - Fully connected layers for classification
    """
    
    def __init__(self, num_classes=100, num_frames=16, input_height=224, input_width=224):
        super(Temporal3DCNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.input_height = input_height
        self.input_width = input_width
        
        # 3D Convolutional blocks
        # Block 1: 3 -> 64 channels
        self.conv_block1 = nn.Sequential(
            Conv3DBlock(3, 64, kernel_size=(3, 3, 3), padding=1),
            Conv3DBlock(64, 64, kernel_size=(3, 3, 3), padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # [64, 8, 112, 112]
        )
        
        # Block 2: 64 -> 128 channels
        self.conv_block2 = nn.Sequential(
            Conv3DBlock(64, 128, kernel_size=(3, 3, 3), padding=1),
            Conv3DBlock(128, 128, kernel_size=(3, 3, 3), padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # [128, 4, 56, 56]
        )
        
        # Block 3: 128 -> 256 channels
        self.conv_block3 = nn.Sequential(
            Conv3DBlock(128, 256, kernel_size=(3, 3, 3), padding=1),
            Conv3DBlock(256, 256, kernel_size=(3, 3, 3), padding=1),
            nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))  # [256, 2, 56, 28]
        )
        
        # Block 4: 256 -> 512 channels
        self.conv_block4 = nn.Sequential(
            Conv3DBlock(256, 512, kernel_size=(3, 3, 3), padding=1),
            Conv3DBlock(512, 512, kernel_size=(3, 3, 3), padding=1),
            nn.MaxPool3d(kernel_size=(2, 1, 2), stride=(2, 1, 2))  # [512, 1, 56, 14]
        )
        
        # Global adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, channels=3, frames, height, width]
               Example: [32, 3, 16, 224, 224]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # Conv blocks
        x = self.conv_block1(x)  # [B, 64, 8, 112, 112]
        x = self.conv_block2(x)  # [B, 128, 4, 56, 56]
        x = self.conv_block3(x)  # [B, 256, 2, 56, 28]
        x = self.conv_block4(x)  # [B, 512, 1, 56, 14]
        
        # Global average pooling
        x = self.adaptive_pool(x)  # [B, 512, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        
        # Fully connected classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # [B, num_classes]
        
        return x


class Temporal3DCNNLite(nn.Module):
    """
    Lightweight 3D CNN for faster training/inference
    
    Suitable for CPU training or quick experiments
    """
    
    def __init__(self, num_classes=100, num_frames=8, input_height=112, input_width=112):
        super(Temporal3DCNNLite, self).__init__()
        
        # Simplified architecture
        self.conv1 = Conv3DBlock(3, 32, kernel_size=(3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv2 = Conv3DBlock(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3 = Conv3DBlock(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, 3, frames, height, width]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
