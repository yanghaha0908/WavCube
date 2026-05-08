import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader  # 新增引用
from tqdm import tqdm
from torch import nn
from vocos import Vocos

def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.extractor = get_vocos_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for VocosWavLMExp: reconstruct test set audios."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="测试集根目录，例如 /inspire/dataset/librispeech/v1/test-clean/",
    )
    parser.add_argument(
        "--config-root",
        type=str,
        default =".",
        # required=True,
        help="Path to Vocos config.yaml",
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        default =".",
        # required=True,
        help="Path to checkpoint(.ckpt) default_config/version_5/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
    )
    # 新增参数，默认为4，这已经能带来很大提升
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of CPU workers for data loading",
    )

    return parser.parse_args()

# 新增 Dataset 类，负责把原来主循环里的加载逻辑挪到后台 Worker 执行
class AudioDataset(Dataset):
    def __init__(self, audio_files, sample_rate):
        self.audio_files = audio_files
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        try:
            # === 原有逻辑：加载与重采样 ===
            audio, sr = torchaudio.load(str(audio_path))

            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                print(f"[WARN] {audio_path} sample rate {sr} != target {self.sample_rate}, resampling.")
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)

            return {
                "audio": audio, 
                "path": str(audio_path)
            }
        except Exception as e:
            print(f"[ERROR] Failed to load {audio_path}: {e}")
            return None

def main():
    args = parse_args()
    
    dataset_path = Path(args.dataset_path)
    config_root = Path(args.config_root)
    ckpt_name = Path(args.ckpt_name)
    sample_rate = args.sample_rate
    
    # config_path = config_root / "config.yaml"
    # ckpt_path = config_root / "checkpoints" / ckpt_name
    # output_path = config_root / "outputs" / ckpt_name.stem
    output_path = Path("/apdcephfs_shfx/share_303787284/sheepgryang/codes/vocos/evaluation/mel_outputs")
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda")

    # 1. 加载模型
    local_path ="/apdcephfs_shfx/share_303787284/sheepgryang/codes/F5-TTS/checkpoints/vocos-mel-24khz"
    print(f"Load vocos from local path {local_path}")
    config_path = f"{local_path}/config.yaml"
    model_path = f"{local_path}/pytorch_model.bin"
    vocoder = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    from vocos.feature_extractors import EncodecFeatures

    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)
    vocoder.load_state_dict(state_dict)
    vocoder = vocoder.eval().to(device)

    # 实例化你刚加进来的 Mel 提取器
    print("[INFO] Initializing MelSpectrogram Extractor...")
    mel_extractor = MelSpec(target_sample_rate=sample_rate).to(device)
    mel_extractor.eval()

    # 2. 准备数据列表 (改为 DataLoader 模式)
    audio_files = list(dataset_path.rglob(f"*.flac"))
    dataset = AudioDataset(audio_files, sample_rate)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    for batch in tqdm(dataloader, desc="Reconstructing"):
        # print(batch["audio"].shape) torch.Size([1, 1, 243040])
        # print(batch["path"]) ['/apdcephfs_sh7/share_302528826/sheepgryang/LibriSpeech/test-clean/8230/279154/8230-279154-0010.flac']

        audio = batch["audio"].squeeze(0).to(device, non_blocking=True)
        path_str = batch["path"][0] # 取出路径字符串
        

        with torch.no_grad():
            gen_mel_spec = mel_extractor(audio)
            recon = vocoder.decode(gen_mel_spec).cpu()
            # recon = vocos(audio)  # shape: [1, T_recon]
        
        # 对齐长度
        target_len = audio.size(1)
        cur_len = recon.size(1)
        if cur_len > target_len:
            recon = recon[:, :target_len]
        elif cur_len < target_len:
            pad_len = target_len - cur_len
            recon = F.pad(recon, (0, pad_len))

        recon = recon.detach().cpu()

        # 3. 构造保存路径，保持和原始数据集相同的层级结构
        audio_path = Path(path_str)
        rel_path = audio_path.relative_to(dataset_path)  # spk/chapter/xxxx.flac
        save_path = output_path / rel_path.with_suffix(".wav")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(str(save_path), recon, sample_rate=sample_rate)
        
    print("[INFO] Done. All reconstructed audios are saved under:")
    print(f"       {output_path}")


if __name__ == "__main__":
    main()