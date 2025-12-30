import torch 
import torch.nn as nn
import torch.nn.functional as F


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


class TemporalAttention(nn.Module):
    def __init__(self, dh_state, eh_state, attn_dim, loc_kernel_size, loc_padding, pos_kernel_size, pos_padding):
        super().__init__()
        self.loc_conv = nn.Conv1d(1, attn_dim, loc_kernel_size, padding=loc_padding)
        self.cum_conv = nn.Conv1d(1, 1, pos_kernel_size, padding=pos_padding)

        self.attn_dim = attn_dim
        self.qproj = nn.Linear(dh_state, attn_dim)
        self.kvproj = nn.Linear(eh_state, attn_dim)

        self.q = nn.Linear(attn_dim, attn_dim)
        self.kv = nn.Linear(attn_dim, attn_dim*2)

    def forward(self, s_t, h_enc, prev_attn, prev_energy):
        prev_attn = prev_attn[:, None, :]
        loc_feats = self.loc_conv(prev_attn).transpose(-2, -1).mean(-1) # (B, T)

        prev_energy = prev_energy[:, None, :]
        cum_feats = self.cum_conv(prev_energy) # (B, 1, T)
        cum_feats = cum_feats.squeeze(1) # (B, T)

        qproj = self.qproj(s_t) # (B, attn_dim)
        kvproj = self.kvproj(h_enc) # (B, T, attn_dim)

        q = self.q(qproj) # (B, attn_dim)

        kv = self.kv(kvproj) # (B, T, attn_dim*2)
        k, v = kv.split(self.attn_dim, dim=-1) # (B, T, attn_dim)

        energy = (q @ k.transpose(-2, -1)) + loc_feats + cum_feats # (B, T)

        a_t = F.softmax(energy, dim=-1)

        context = a_t @ v # (B, attn_dim)

        return context, a_t, energy


'''
Decoder RNN (2-layer LSTM, 1024 hidden):
Input per step: [prenet(prev_mel), context_t] → [B, 256 + attn_dim]
Output: LSTM hidden → project to mel_dim (80)

Autoregressive loop ~max_mel_frames:
1. Prenet(previous mel frame or <sos>)
2. Attention(s_t=decoder_hidden, h_enc, prev_attn, prev_energy) → context, a_t, energy
3. Concat(prenet_out, context) → decoder LSTM input
4. LSTM → new s_{t+1}
5. Linear(mel_dim) → mel_pred
'''