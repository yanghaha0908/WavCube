from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        max_retries = 5  # 最大重试次数，防止卡住
        retries = 0
        while retries < max_retries:
            audio_path = self.filelist[index]
            audio_path = audio_path.replace("/apdcephfs_sh7/share_302528826","/apdcephfs_tj5/share_303787284")
            try:
                y, sr = torchaudio.load(audio_path)
                if y.size(0) > 1:
                    # mix to mono
                    y = y.mean(dim=0, keepdim=True)
                gain = np.random.uniform(-1, -6) if self.train else -3
                y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
                if sr != self.sampling_rate:
                    y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
                if y.size(-1) < self.num_samples:
                    pad_length = self.num_samples - y.size(-1)
                    padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
                    y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
                elif self.train:
                    start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
                    y = y[:, start : start + self.num_samples]
                else:
                    # During validation, take always the first segment for determinism
                    y = y[:, : self.num_samples]
            
                return y[0]
            except Exception as e:
                print(f"[WARN] Failed to load {audio_path}: {e}")
                retries += 1
                index = (index + 1) % len(self.filelist)
                if retries == max_retries:
                    raise RuntimeError(f"Failed to load a valid audio file after {max_retries} retries.")
        
        return torch.zeros(1, self.num_samples*3)

class VocosEmiliaDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosEmiliaDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosEmiliaDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        self.jsonl_path = cfg.filelist_path
        self.idx_path = self.jsonl_path+".idx"
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        self.libriheavy_root = "/apdcephfs_tj5/share_303787284/sheepgryang/data/libriheavy" #new
        self.offsets = []
        print(f"Loading offsets from {self.idx_path}...")
        with open(self.idx_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                self.offsets.append(int(line.strip()))
    
        self.jsonl_file = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _get_file_handle(self):
        """获取当前进程的文件句柄，如果不存在则创建"""
        if self.jsonl_file is None:
            self.jsonl_file = open(self.jsonl_path, 'r', encoding='utf-8')
        return self.jsonl_file

    def _read_jsonl_line(self, index):
        """根据 offset 读取并解析 jsonl"""
        offset = self.offsets[index]
        f = self._get_file_handle()
        f.seek(offset)
        line = f.readline().strip()
        return line

    def __getitem__(self, index: int) -> torch.Tensor:
        max_retries = 5  # 最大重试次数，防止卡住
        retries = 0
        while retries < max_retries:
            line = self._read_jsonl_line(index)
            try:
                if line.startswith('{'):
                    # --- Libriheavy 格式 ---
                    data = json.loads(line)
                    rel_path = data['recording']['sources'][0]['source']
                    audio_path = os.path.join(self.libriheavy_root, rel_path)
                    start_sec = data['start']
                    duration_sec = data['duration']
                else:
                    # --- LibriTTS,Emilia 格式 ---
                    audio_path = line
                    audio_path = audio_path.replace("/apdcephfs_sh7/share_302528826","/apdcephfs_tj5/share_303787284")
                    start_sec = 0.0
                    duration_sec = None  # 读全长                

                # 2. 高效读取片段                
                if duration_sec is not None:
                    start_frame = int(start_sec * self.sampling_rate)
                    num_frames = int(duration_sec * self.sampling_rate)
                    y, sr = torchaudio.load(audio_path, frame_offset=start_frame, num_frames=num_frames)
                else:
                    y, sr = torchaudio.load(audio_path)

                if y.size(0) > 1:
                    # mix to mono
                    y = y.mean(dim=0, keepdim=True)
                gain = np.random.uniform(-1, -6) if self.train else -3
                y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
                if sr != self.sampling_rate:
                    y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
                if y.size(-1) < self.num_samples:
                    pad_length = self.num_samples - y.size(-1)
                    padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
                    y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
                elif self.train:
                    start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
                    y = y[:, start : start + self.num_samples]
                else:
                    # During validation, take always the first segment for determinism
                    y = y[:, : self.num_samples]
            
                return y[0]
            except Exception as e:
                print(f"[WARN] Failed to load {audio_path}: {e}")
                retries += 1
                index = (index + 1) % len(self.offsets)  # 移动到下一个音频文件
                if retries == max_retries:
                    raise RuntimeError(f"Failed to load a valid audio file after {max_retries} retries.")
        
        return torch.zeros(1, self.num_samples*3)
