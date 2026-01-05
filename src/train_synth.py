import os
import csv
import random
from typing import List, Tuple, Optional

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

sample_rate = 16_000
n_mels = 40
win_length = int(0.025 * sample_rate)
hop_length = int(0.010 * sample_rate)
n_fft = 512


def wav_to_mel(
    file_path: str,
    sample_rate: int = sample_rate,
    n_mels: int = n_mels,
    win_length: int = win_length,
    hop_length: int = hop_length,
    n_fft: int = n_fft,
) -> Optional[torch.Tensor]:
    try:
        waveform, sr = torchaudio.load(file_path)

        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # mixdown to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        mel = mel_spec(waveform)       # (1, n_mels, T)
        return mel.squeeze(0)          # (n_mels, T)

    except Exception:
        return None


symbols = [
    " ", "!", "'", "(", ")", ",", "-", ".", ":", ";", "?",  # punctuation
] + [chr(i) for i in range(ord("a"), ord("z") + 1)]         # a–z

symbol_to_id = {s: i for i, s in enumerate(symbols)}


def text_to_ids(text: str) -> torch.Tensor:
    text = text.lower()
    ids = [symbol_to_id[c] for c in text if c in symbol_to_id]
    if len(ids) == 0:
        # avoid empty sequences
        ids = [symbol_to_id[" "]]
    return torch.tensor(ids, dtype=torch.long)


class LJSpeechDataset(Dataset):
    def __init__(
        self,
        root: str,
        use_normalized: bool = True,
        mode: str = "full",
        target_frames: Optional[int] = None,
    ):
        assert mode in ("full", "train", "eval")
        if mode in ("train", "eval"):
            assert target_frames is not None, "target_frames must be set for train/eval mode"

        self.data_root = os.path.join(root, "LJSpeech-1.1")
        self.wav_dir = os.path.join(self.data_root, "wavs")
        self.metadata_path = os.path.join(self.data_root, "metadata.csv")

        self.use_normalized = use_normalized
        self.mode = mode
        self.target_frames = target_frames

        self.items: List[Tuple[str, str]] = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                # Skip empty / malformed lines
                if len(row) < 2:
                    continue

                file_id = row[0].strip()
                # LJSpeech: [ID, Transcription, Normalized Transcription][web:2][web:3]
                if self.use_normalized and len(row) >= 3:
                    text = row[2].strip()
                else:
                    text = row[1].strip()

                if not file_id:
                    continue

                wav_path = os.path.join(self.wav_dir, file_id + ".wav")
                if os.path.exists(wav_path):
                    self.items.append((wav_path, text))

        if len(self.items) == 0:
            raise RuntimeError(f"No valid items found in metadata at {self.metadata_path}")

        print(f"LJSpeechDataset: loaded {len(self.items)} utterances from {self.metadata_path}")

    def __len__(self) -> int:
        return len(self.items)

    def _random_crop_or_pad(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (n_mels, T)
        _, frames = mel.shape
        T = self.target_frames

        if frames > T:
            start = random.randint(0, frames - T)
            mel = mel[:, start:start + T]
        elif frames < T:
            pad = T - frames
            mel = F.pad(mel, (0, pad))
        return mel

    def _center_crop_or_pad(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (n_mels, T)
        _, frames = mel.shape
        T = self.target_frames

        if frames > T:
            start = (frames - T) // 2
            mel = mel[:, start:start + T]
        elif frames < T:
            pad = T - frames
            mel = F.pad(mel, (0, pad))
        return mel

    def __getitem__(self, idx: int):
        wav_path, text = self.items[idx]
        mel = wav_to_mel(wav_path)  # (n_mels, T)

        if mel is None:
            # basic fallback
            if self.mode == "full":
                mel = torch.zeros(n_mels, 1)
            else:
                mel = torch.zeros(n_mels, self.target_frames)

        if self.mode == "train":
            mel = self._random_crop_or_pad(mel)   # (n_mels, target_frames)
        elif self.mode == "eval":
            mel = self._center_crop_or_pad(mel)   # (n_mels, target_frames)
        # mode == "full": keep original length

        mel_TF = mel.T   # (T, n_mels)

        return mel_TF, text, wav_path



def ljspeech_collate_fn(batch):
    mels, texts, wav_paths = zip(*batch)

    # ------- mel padding -------
    mel_lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    max_mel_len = int(mel_lengths.max().item())
    B = len(mels)
    n_mels_local = mels[0].shape[1]

    mel_padded = torch.zeros(B, max_mel_len, n_mels_local)
    for i, mel in enumerate(mels):
        T = mel.shape[0]
        mel_padded[i, :T] = mel

    text_ids_list = [text_to_ids(t) for t in texts]
    text_lengths = torch.tensor([len(ti) for ti in text_ids_list], dtype=torch.long)
    max_text_len = int(text_lengths.max().item())

    text_padded = torch.zeros(B, max_text_len, dtype=torch.long)
    for i, ti in enumerate(text_ids_list):
        text_padded[i, : len(ti)] = ti

    return {
        "mels": mel_padded,          # (B, T_mel, n_mels)
        "mel_lengths": mel_lengths,
        "text_ids": text_padded,     # (B, T_text)
        "text_lengths": text_lengths,
        "wav_paths": list(wav_paths),
    }


root = "data"  


ds_full = LJSpeechDataset(root=root, mode="full", use_normalized=True)
dl_full = DataLoader(ds_full, batch_size=8, shuffle=True, collate_fn=ljspeech_collate_fn)

