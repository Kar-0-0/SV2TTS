import torch 
import torch.nn.functional as F
import torchaudio
import os
from torch.utils.data import Dataset
import random
from pathlib import Path
from speaker_encoder import SpeakerEncoder, GE2ELoss



file_path = 'data/wav/id10001/1zcIwhmdeo4/00001.wav'
sample_rate = 16_000
n_mels = 40
win_length = int(0.025 * sample_rate)
hop_length= int(0.010 * sample_rate)
n_fft = 512
hidden_channels = 768
lstm_layers = 3
n_emb = 256
epochs = 100_000
n_speakers = 32
n_utter = 10
learning_rate = 1e-4



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
        
        # Crop or pad to target_frames (RANDOM for training)
        mel = self._adjust_length(mel)
        
        return mel.T  # Return [160, 40]
    
    def _adjust_length(self, mel):
        # RANDOM crop for training
        _, frames = mel.shape
        
        if frames > self.target_frames:
            start = random.randint(0, frames - self.target_frames)
            mel = mel[:, start:start + self.target_frames]
        elif frames < self.target_frames:
            pad = self.target_frames - frames
            mel = torch.nn.functional.pad(mel, (0, pad))
        
        return mel

    def _adjust_length_eval(self, mel):
        # DETERMINISTIC crop for evaluation
        _, frames = mel.shape
        
        if frames > self.target_frames:
            # center crop
            start = (frames - self.target_frames) // 2
            mel = mel[:, start:start + self.target_frames]
        elif frames < self.target_frames:
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
model.load_state_dict(torch.load("checkpoint_step2000.pth"))


loss_fn = GE2ELoss()


optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 1e-4},
    {'params': loss_fn.parameters(), 'lr': 1e-5}
])


device = torch.device('cpu')


def verify_speakers(model, audio1_path, audio2_path, threshold=0.7):
    model.eval()
    with torch.no_grad():
        # Load and convert both audio files
        mel1 = wav_to_mel(audio1_path, sample_rate, n_mels, win_length, hop_length, n_fft)
        mel2 = wav_to_mel(audio2_path, sample_rate, n_mels, win_length, hop_length, n_fft)
        
        # Check if files loaded successfully
        if mel1 is None or mel2 is None:
            print("Error: Could not load one or both audio files")
            return False
        
        # Adjust length and prepare (DETERMINISTIC for eval)
        mel1 = dataset._adjust_length_eval(mel1).T.unsqueeze(0)  # [1, 160, 40]
        mel2 = dataset._adjust_length_eval(mel2).T.unsqueeze(0)  # [1, 160, 40]
        
        # Get embeddings
        emb1 = model(mel1)  # [1, 256]
        emb2 = model(mel2)  # [1, 256]
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(emb1, emb2).item()
        
        print(f"Similarity: {similarity:.4f}")
        return similarity > threshold
    

model.train()
for epoch in range(epochs):
    x_batch = dataset.sample_batch(n_speakers, n_utter)  # [64, 10, 160, 40]

    N, M, T, Z = x_batch.shape
    x_batch = x_batch.to(device)
    x_batch = x_batch.view(N * M, T, Z) 
    for i in range(N * M):
        mean = x_batch[i].mean()
        std = x_batch[i].std() + 1e-8
        x_batch[i] = (x_batch[i] - mean) / std

    if epoch == 0:
        print(f"Normalized batch - mean: {x_batch.mean():.4f}, std: {x_batch.std():.4f}")
        
    embeddings = model(x_batch)  # [640, 256]
    embeddings = embeddings.view(N, M, -1)
    
    loss = loss_fn(embeddings)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
    torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), max_norm=0.5)
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Step {2_000 + epoch}: Loss {loss.item():.4f}, w={loss_fn.w.item():.2f}, b={loss_fn.b.item():.2f}\n------------------------")
        print(f"  Embedding sample: {embeddings[0, 0, :5]}")    
        with torch.no_grad():
            model.eval()
            emb_norm = torch.norm(embeddings, dim=-1).mean()
            emb_std = embeddings.std()
            print(f"  Embedding norm: {emb_norm:.4f}, std: {emb_std:.4f}")

            # Test with same speaker
            same_speaker = verify_speakers(model, 'data/wav/id11161/9-b6cfguVgI/00001.wav', 
                                                'data/wav/id11161/zCoxj-QjqbE/00001.wav')
            print(f"Same speaker: {same_speaker}")

            # Test with different speakers
            diff_speaker = verify_speakers(model, 'data/wav/id11240/cdZCw06aRE8/00001.wav',
                                                'data/wav/id10989/SwqPzS1-aoc/00001.wav')
            print(f"Different speaker: {diff_speaker}")
            print("------------------------")   
        model.train()

    if epoch % 1000 == 0 and epoch > 0:
        torch.save(model.state_dict(), f'checkpoint_step{2_000 + epoch}.pth')

torch.save(model.state_dict(), f'se_step_final.pth')
