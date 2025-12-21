import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    def __init__(self, mel_bins, hidden_channels, lstm_layers, n_emb, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(mel_bins, hidden_channels, lstm_layers, batch_first=batch_first)
        self.proj = nn.Linear(hidden_channels, n_emb)
    
    def forward(self, x):
        x, _ = self.lstm(x) # (B, N, hidden_channels)
        x = x[:, -1, :] # (B, hidden_channels)
        x = self.proj(x) # (B, n_emb)
        x = F.normalize(x, dim=-1)

        return x


class GE2ELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))
    
    def forward(self, x):
        N, M, D = x.shape

        x_flat = x.view(N * M, D)

        centroids_full = torch.mean(x, dim=1) # (N, D)
        centroids_expanded = centroids_full[:, None, :].expand(N, M, D)

        centroids_excl = (M * centroids_expanded - x) / (M - 1)  # (N, M, D)
        centroids_excl = centroids_excl.view(N * M, D)  # (N*M, D)
        centroids_incl_expanded = centroids_full[None, :, :].expand(N*M, N, D)
        speaker_ids = torch.arange(N).repeat_interleave(M)  # [0,0,1,1,2,2,...]
        row_indices = torch.arange(N * M)
        centroids_final = centroids_incl_expanded.clone()
        centroids_final[row_indices, speaker_ids] = centroids_excl

        cos_sim = torch.sum(x_flat[:, None, :] * centroids_final, dim=2) # (N*M, N)

        cos_sim = self.w * cos_sim + self.b

        targets = speaker_ids
        loss = F.cross_entropy(cos_sim, targets)

        return loss


loss_fn = GE2ELoss()
# Perfect embeddings: all utterances from same speaker are identical
perfect_emb = torch.randn(4, 1, 256)  # one random vector per speaker
perfect_emb = perfect_emb.repeat(1, 3, 1)  # repeat 3 times per speaker
perfect_emb = F.normalize(perfect_emb, p=2, dim=-1)

loss = loss_fn(perfect_emb)
print(loss)  # should be very small, like 0.01-0.1