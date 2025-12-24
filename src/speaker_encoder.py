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
        B, N, M, D = x.shape
        
        # Flatten to (B, N*M, D)
        x_flat = x.view(B, N * M, D)
        x_flat = F.normalize(x_flat, p=2, dim=-1)  # L2 normalize
        
        centroids_full = torch.mean(x, dim=2)
        centroids_full = F.normalize(centroids_full, p=2, dim=-1)
        
        # Expand to (B, N, M, D)
        centroids_expanded = centroids_full[:, :, None, :].expand(B, N, M, D)
        
        # Compute exclusion centroids (B, N, M, D)
        centroids_excl = (M * centroids_expanded - x) / (M - 1)
        centroids_excl = centroids_excl.view(B, N * M, D)  # (B, N*M, D)
        centroids_excl = F.normalize(centroids_excl, p=2, dim=-1)
        
        # Expand inclusion centroids to (B, N*M, N, D)
        centroids_incl_expanded = centroids_full[:, None, :, :].expand(B, N*M, N, D)
        
        # Create indices for each batch
        speaker_ids = torch.arange(N, device=x.device).repeat_interleave(M)  # [0,0,1,1,2,2,...]
        row_indices = torch.arange(N * M, device=x.device)
        
        # Replace own-speaker centroids with exclusion centroids
        centroids_final = centroids_incl_expanded.clone()
        for b in range(B):
            centroids_final[b, row_indices, speaker_ids] = centroids_excl[b]
        
        # Compute cosine similarity (B, N*M, N)
        cos_sim = torch.sum(x_flat[:, :, None, :] * centroids_final, dim=3)
        
        # Scale and bias
        cos_sim = self.w * cos_sim + self.b
        
        # Compute loss for each batch sample
        targets = speaker_ids.expand(B, -1)  # (B, N*M)
        loss = F.cross_entropy(cos_sim.view(B * N * M, N), targets.reshape(-1))
        
        return loss
