import torch 
import torch.nn as nn
import torch.nn.functional as F

# Architecture
'''
1. a recurrent sequence-to-sequence feature prediction network with
attention which predicts a sequence of mel spectrogram frames from
an input character sequence

2. a modified version of WaveNet which generates time-domain waveform
samples conditioned on the predicted mel spectrogram frames.
'''

# Encoder
    # BiLSTM
    # decoder hidden state at the previous time step as the query,
    # and the encoder hidden states at all time steps as both the
    # keys and values
# Decoder
    # Attention - Location Based Attention or Bahdanau Attention


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            n_emb,
            n_channels,
            n_filters,
            kernel_size, 
            hidden_channels,
            lstm_layers,
            lstm_in
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_emb)

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_filters, kernel_size=kernel_size, padding=(2, 0))
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=(2, 0))
        self.bn2 = nn.BatchNorm2d(n_filters)

        self.conv3 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=(2, 0))
        self.bn3 = nn.BatchNorm2d(n_filters)
        
        self.relu = nn.ReLU()

        self.bi_lstm = nn.LSTM(lstm_in, hidden_channels, lstm_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.emb(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        B, C, T, D = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, C*D)

        x = self.bi_lstm(x)

        return x
