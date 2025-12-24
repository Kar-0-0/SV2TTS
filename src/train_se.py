import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
from speaker_encoder import SpeakerEncoder, GE2ELoss


file_path = 'data/wav/id10001/1zcIwhmdeo4/00001.wav'
sample_rate = 16_000
n_mels = 40
win_length = int(0.025 * sample_rate)
hop_length= int(0.010 * sample_rate)
n_fft = 512
batch_size = 32
num_workers = 2
hidden_channels = 768
lstm_layers = 3
n_emb = 256
epochs = 100
n_speakers = 64
n_utter = 10
learning_rate = 1e-3


# load data
def wav_to_mel(file_path, sample_rate, n_mels, win_length, hop_length, n_fft):
    try:
        waveform, sr = torchaudio.load(file_path)
        
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
        mel = mel_spec(waveform)
        
        return mel.squeeze(0)
    
    except Exception as e:
        # Return None for corrupted files
        return None


class VoxCelebDataset(Dataset):
    def __init__(self, data_dir, wav_to_mel_fn, target_frames=160):
        self.data_dir = data_dir
        self.wav_to_mel_fn = wav_to_mel_fn
        self.target_frames = target_frames
        
        # Build speaker_id -> list of file paths mapping
        self.speaker_to_files = {}
        self.all_files = []
        
        corrupted_count = 0
        
        for speaker_id in os.listdir(data_dir):
            speaker_path = os.path.join(data_dir, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
            
            files = []
            # Walk through video_id folders
            for video_id in os.listdir(speaker_path):
                video_path = os.path.join(speaker_path, video_id)
                if not os.path.isdir(video_path):
                    continue
                
                # Collect all .wav files
                for wav_file in os.listdir(video_path):
                    if wav_file.endswith('.wav'):
                        full_path = os.path.join(video_path, wav_file)
                        # Filter out empty/corrupted files
                        if os.path.getsize(full_path) > 1000:  # At least 1KB
                            files.append(full_path)
                        else:
                            corrupted_count += 1
            
            if len(files) > 0:
                self.speaker_to_files[speaker_id] = files
                self.all_files.extend(files)
        
        self.speaker_ids = list(self.speaker_to_files.keys())
        print(f"Loaded {len(self.speaker_ids)} speakers with {len(self.all_files)} utterances")
        print(f"Skipped {corrupted_count} corrupted files")
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        mel = self.wav_to_mel_fn(file_path)  # [40, frames]
        
        if mel is None:
            # Return zeros if corrupted
            return torch.zeros(self.target_frames, 40)
        
        # Crop or pad to target_frames
        mel = self._adjust_length(mel)
        
        return mel.T  # Return [160, 40]
    
    def _adjust_length(self, mel):
        _, frames = mel.shape
        
        if frames > self.target_frames:
            # Randomly crop
            start = random.randint(0, frames - self.target_frames)
            mel = mel[:, start:start + self.target_frames]
        elif frames < self.target_frames:
            # Pad with zeros
            pad = self.target_frames - frames
            mel = torch.nn.functional.pad(mel, (0, pad))
        
        return mel
    
    def sample_batch(self, N, M):
        batch = []
        
        # Sample N random speakers
        sampled_speakers = random.sample(self.speaker_ids, N)
        
        for speaker_id in sampled_speakers:
            files = self.speaker_to_files[speaker_id]
            speaker_mels = []
            
            # Keep trying until we get M valid utterances
            attempts = 0
            max_attempts = M * 3  # Try up to 3x
            
            while len(speaker_mels) < M and attempts < max_attempts:
                file_path = random.choice(files)
                mel = self.wav_to_mel_fn(file_path)
                
                if mel is not None:
                    mel = self._adjust_length(mel)
                    speaker_mels.append(mel.T)  # [160, 40]
                
                attempts += 1
            
            # If we still don't have enough, pad with zeros
            while len(speaker_mels) < M:
                speaker_mels.append(torch.zeros(self.target_frames, 40))
            
            batch.append(torch.stack(speaker_mels[:M]))  # [M, 160, 40]
        
        return torch.stack(batch)  # [N, M, 160, 40]


dataset = VoxCelebDataset(
    data_dir='data/wav',
    wav_to_mel_fn=lambda path: wav_to_mel(path, sample_rate, n_mels, win_length, hop_length, n_fft)
)

model = SpeakerEncoder(
    mel_bins=n_mels,
    hidden_channels=hidden_channels,
    lstm_layers=lstm_layers,
    n_emb=n_emb 
)

loss_fn = GE2ELoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')

for epoch in range(epochs):
    x_batch = dataset.sample_batch(n_speakers, n_utter)  # [64, 10, 160, 40]
    x_batch = x_batch.to(device)
    
    # Reshape to process all utterances through model
    N, M, T, F = n_speakers, n_utter, 160, n_mels
    x_batch = x_batch.view(N * M, T, F) 
    
    embeddings = model(x_batch)  # [640, 256]
    embeddings = embeddings.view(1, N, M, -1)
    loss = loss_fn(embeddings)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: {loss.item():.4f}")




