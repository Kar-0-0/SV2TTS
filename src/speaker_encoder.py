import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    def __init__(self, mel_bins, hidden_channels, lstm_layers, n_emb, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(mel_bins, hidden_channels, lstm_layers, 
                           batch_first=batch_first, dropout=0.0)
        self.proj = nn.Linear(hidden_channels, n_emb)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.step_counter = 0
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        
        if self.training:
            self.step_counter += 1
            if self.step_counter % 100 == 1:
                print(f"  After LSTM - std: {x.std().item():.4f}")
        
        x = self.proj(x)
        
        if self.training and self.step_counter % 100 == 1:
            print(f"  After projection - std: {x.std().item():.4f}")
        
        x = F.normalize(x, dim=-1)
        
        if self.training and self.step_counter % 100 == 1:
            print(f"  After normalize - std: {x.std().item():.4f}")

        return x


class GE2ELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(15.0))
        self.b = nn.Parameter(torch.tensor(-7.0))
    
    def forward(self, x):
        # x: (N, M, D) - N speakers, M utterances, D embedding dim
        N, M, D = x.shape
        
        centroids_incl = torch.mean(x, dim=1) # (N, D)
        

        centroids_excl = (M * centroids_incl[:, None, :] - x) / (M - 1)
        
        # Flatten: (N*M, D)
        x_flat = x.view(N * M, D)
        centroids_excl_flat = centroids_excl.view(N * M, D)

        sim_matrix = torch.matmul(x_flat, centroids_incl.T)  # (N*M, N)
        

        for j in range(N):
            start_idx = j * M
            end_idx = start_idx + M
            sim_matrix[start_idx:end_idx, j] = torch.sum(
                x_flat[start_idx:end_idx] * centroids_excl_flat[start_idx:end_idx],
                dim=1
            )
        
        sim_matrix = self.w * sim_matrix + self.b
        
        labels = torch.arange(N, device=x.device).repeat_interleave(M)
        
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
