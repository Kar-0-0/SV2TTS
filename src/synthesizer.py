import torch 
import torch.nn as nn
import torch.nn.functional as F
from speaker_encoder import SpeakerEncoder


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
            lstm_in,
            batch_first=True,
            bidirectional=False
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

        self.bi_lstm = nn.LSTM(lstm_in, hidden_channels, lstm_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x):
        x = self.emb(x).transpose(1, 2) # (B, n_emb, T)
        x = x[:, None, :, :] # (B, 1, n_emb, T)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        B, C, D, T = x.shape
        x = x.transpose(1, 3).contiguous().view(B, T, C*D)

        x, _ = self.bi_lstm(x)

        return x # (B, T, hidden_channels*2)


class TemporalAttention(nn.Module):
    def __init__(
            self,
            lstm_dim,
            eh_state,
            attn_dim,
            loc_kernel_size,
            loc_padding,
            pos_kernel_size,
            pos_padding
    ):
        super().__init__()
        self.loc_conv = nn.Conv1d(1, attn_dim, loc_kernel_size, padding=loc_padding)
        self.cum_conv = nn.Conv1d(1, 1, pos_kernel_size, padding=pos_padding)

        self.attn_dim = attn_dim
        self.qproj = nn.Linear(lstm_dim, attn_dim)
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

        energy = torch.einsum("bd,btd->bt", q, k) # (B, T)
        energy = energy + loc_feats + cum_feats # (B, T)

        a_t = F.softmax(energy, dim=-1)

        context = torch.einsum("bt,btd->bd", a_t, v) # (B, attn_dim)

        return context, a_t, energy


class PreNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = nn.Linear(in_channels, out_channels)
        self.l2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))

        return x # (B, out_channels)


class PostNet(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            kernel_size,
            padding,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(n_filters)

        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(n_filters)

        self.conv4 = nn.Conv2d(n_filters, n_filters, kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(n_filters)

        self.conv5 = nn.Conv2d(n_filters, in_channels, kernel_size, padding=padding)
        self.bn5 = nn.BatchNorm2d(in_channels)

        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.tanh(self.bn1(self.conv1(x)))
        x = self.tanh(self.bn2(self.conv2(x)))
        x = self.tanh(self.bn3(self.conv3(x)))
        x = self.tanh(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        return x


class Decoder(nn.Module):
    def __init__(
            self,
            pre_in,
            pre_out,
            eh_state,
            attn_dim,
            loc_kernel_size,
            loc_padding,
            pos_kernel_size,
            pos_padding,
            lstm_dim,
            lstm_layers,
            mel_dim,
            post_in,
            post_filters,
            post_kernel,
            post_padding,
            batch_first=True,
            bidirectional=False
    ):
        super().__init__()
        self.lstm_layers = lstm_layers
        self.lstm_dim = lstm_dim
        self.mel_dim = mel_dim

        self.pre_net = PreNet(pre_in, pre_out)

        self.temp_attn = TemporalAttention(
            lstm_dim,
            eh_state,
            attn_dim,
            loc_kernel_size,
            loc_padding,
            pos_kernel_size,
            pos_padding
        )

        self.lstm = nn.LSTM(pre_out+attn_dim, lstm_dim, lstm_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.mel_proj = nn.Linear(lstm_dim, mel_dim)

        self.post_net = PostNet(
            post_in,
            post_filters,
            post_kernel,
            post_padding,
            mel_dim
        )

        self.gate_proj = nn.Linear(lstm_dim, 1)

    def forward(self, x, mel_seq):
        device = x.device
        B, t_enc, _ = x.shape
        mel_out = torch.zeros_like(mel_seq, device=device)

        gate_out = []

        for t in range(mel_seq.size(1)):
            if t == 0:
                h = torch.zeros(self.lstm_layers, B, self.lstm_dim, device=device)
                c = torch.zeros(self.lstm_layers, B, self.lstm_dim, device=device)
                s_t = h[-1]
                h_enc = x

                prev_attn = torch.ones((B, t_enc), device=device)
                prev_attn = F.normalize(prev_attn, p=1, dim=-1) # uniform

                prev_energy = torch.zeros((B, t_enc), device=device)

            context, a_t, energy = self.temp_attn(s_t, h_enc, prev_attn, prev_energy)
            prev_attn = a_t
            prev_energy = energy
            
            mel_frame = mel_seq[:, t]
            pre_out = self.pre_net(mel_frame) # (B, pre_out)
            lstm_in = torch.cat([context, pre_out], dim=-1) # (B, attn_dim+pre_out)
            lstm_in = lstm_in[:, None, :] # (B, 1, attn_dim+pre_out)

            out, (h, c) = self.lstm(lstm_in, (h, c)) # out: (B, 1, lstm_dim)
            s_t = out[:, 0] # (B, lstm_dim)
            mel_frame = self.mel_proj(s_t) # (B, mel_dim)
            gate_t = self.gate_proj(s_t).squeeze(-1)

            mel_out[:, t] = mel_frame
            gate_out.append(gate_t.unsqueeze(1))
        
        gate_out = torch.cat(gate_out, dim=1)
        
        mel_4D = mel_out[:, None, :, :].transpose(-2, -1) # (B, 1, mel_dim, T)
        mel_post_4d = self.post_net(mel_4D) # (B, 1, mel_dim, T)
        mel_post = mel_post_4d.squeeze(1).transpose(1, 2) # (B, T, mel_dim)

        mel_out_post = mel_out + mel_post

        return mel_out, mel_out_post, gate_out

    def generate(self, max_new_frames, voice_embedding, synth_encoder, text_ids, device=None):
        h_enc = synth_encoder(text_ids).to(device)
        h_enc = torch.cat([h_enc, voice_embedding], dim=-1) # (B, T, enc_dim + voice_dim)

        B, t_enc, _ = h_enc.shape

        h = torch.zeros(self.lstm_layers, B, self.lstm_dim, device=device)
        c = torch.zeros(self.lstm_layers, B, self.lstm_dim, device=device)
        s_t = h[-1]

        prev_attn = torch.ones(B, t_enc, device=device)
        prev_attn = F.normalize(prev_attn, p=1, dim=-1)
        prev_energy = torch.zeros(B, t_enc, device=device)

        prev_mel = torch.zeros(B, self.mel_dim, device=device)

        mel_frames = []

        for _ in range(max_new_frames):
            context, a_t, energy = self.temp_attn(s_t, h_enc, prev_attn, prev_energy)
            prev_attn, prev_energy = a_t, energy

            pre_out = self.pre_net(prev_mel) # (B, pre_out)
            lstm_in = torch.cat([context, pre_out], dim=-1) # (B, attn_dim + pre_out)
            lstm_in = lstm_in.unsqueeze(1)  

            out, (h, c) = self.lstm(lstm_in, (h, c)) # out: (B, 1, lstm_dim)
            s_t = out[:, 0] # (B, lstm_dim)

            mel_t = self.mel_proj(s_t) # (B, mel_dim)
            mel_frames.append(mel_t.unsqueeze(1))

            prev_mel = mel_t
            
            stop_token = self.gate_proj(s_t).squeeze(-1) # (B,)
            stop_prob = torch.sigmoid(stop_token)

            gate_threshold = 0.5
            if (stop_prob > gate_threshold).all():
                break

        mel_out = torch.cat(mel_frames, dim=1) # (B, T_dec, mel_dim)

        mel_4D = mel_out.transpose(1, 2).unsqueeze(1) # (B, 1, mel_dim, T_dec)
        mel_post_4d = self.post_net(mel_4D) # (B, 1, mel_dim, T_dec)
        mel_post = mel_post_4d.squeeze(1).transpose(1, 2) # (B, T_dec, mel_dim)

        mel_out_post = mel_out + mel_post

        return mel_out, mel_out_post

