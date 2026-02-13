import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLSTM(nn.Module):
    """LSTM-based model for sign language recognition from MediaPipe landmarks
    
    Architecture:
    - Input: [batch_size, frames, features] = [batch, 48, 225]
    - 1-layer bidirectional LSTM with 128 hidden units (optimized for CPU)
    - Global average pooling over temporal dimension
    - Fully connected layers with dropout
    - Output: [batch_size, num_classes]
    """
    
    def __init__(self, input_features=225, num_classes=100, hidden_size=128, 
                 num_layers=1, dropout=0.5, bidirectional=True):
        super(TemporalLSTM, self).__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer (bidirectional) - optimized for CPU
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # LSTM output size (doubled if bidirectional)
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Fully connected layers - simplified
        self.fc_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, frames, features]
               Example: [32, 48, 225] for batch_size=32, frames=48, features=225
        
        Returns:
            logits: Output tensor of shape [batch_size, num_classes]
        """
        # x shape: [batch_size, frames, features]
        
        # LSTM forward pass
        # lstm_out: [batch_size, frames, hidden_size*num_directions]
        # (h_n, c_n): final hidden and cell states
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Method 1: Use last frame's hidden state (simpler, faster)
        # last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_size*num_directions]
        
        # Method 2: Global average pooling over all frames (better temporal modeling)
        # Aggregate information from all 48 frames
        x = torch.mean(lstm_out, dim=1)  # [batch_size, hidden_size*num_directions]
        
        # Fully connected classifier
        x = self.fc_dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)  # [batch_size, num_classes]
        
        return x


class TemporalGRU(nn.Module):
    """GRU-based model for sign language recognition (faster LSTM alternative)
    
    Similar to LSTM but with fewer parameters and slightly faster computation.
    """
    
    def __init__(self, input_features=225, num_classes=100, hidden_size=128,
                 num_layers=1, dropout=0.5, bidirectional=True):
        super(TemporalGRU, self).__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU layer (bidirectional) - optimized for CPU
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # GRU output size (doubled if bidirectional)
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Fully connected layers - simplified
        self.fc_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(gru_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, frames, features]
        
        Returns:
            logits: Output tensor of shape [batch_size, num_classes]
        """
        # GRU forward pass
        gru_out, h_n = self.gru(x)
        
        # Global average pooling over temporal dimension
        x = torch.mean(gru_out, dim=1)  # [batch_size, hidden_size*num_directions]
        
        # Fully connected classifier
        x = self.fc_dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)
        
        return x
