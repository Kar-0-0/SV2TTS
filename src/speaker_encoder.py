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
        self.w = nn.Parameter(torch.tensor(15.0))
        self.b = nn.Parameter(torch.tensor(-7.0))
    
    def forward(self, x):
        # x: (N, M, D) - N speakers, M utterances, D embedding dim
        N, M, D = x.shape
        
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=-1)
        
        # Compute centroids including all utterances: (N, D)
        centroids_incl = torch.mean(x, dim=1)
        
        # Compute exclusion centroids for each utterance
        # For each utterance, exclude it from its speaker's centroid
        centroids_excl = (M * centroids_incl[:, None, :] - x) / (M - 1)
        
        # Flatten: (N*M, D)
        x_flat = x.view(N * M, D)
        centroids_excl_flat = centroids_excl.view(N * M, D)
        
        # Compute similarity matrix: (N*M, N)
        # For each utterance, compute similarity to all N centroids
        sim_matrix = torch.matmul(x_flat, centroids_incl.T)  # (N*M, N)
        
        # Replace diagonal blocks with exclusion centroids
        for j in range(N):
            start_idx = j * M
            end_idx = start_idx + M
            sim_matrix[start_idx:end_idx, j] = torch.sum(
                x_flat[start_idx:end_idx] * centroids_excl_flat[start_idx:end_idx],
                dim=1
            )
        
        # Scale and shift
        sim_matrix = self.w * sim_matrix + self.b
        
        # Create labels: each utterance belongs to its speaker
        labels = torch.arange(N, device=x.device).repeat_interleave(M)
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
