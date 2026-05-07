"""
示例一：从音频提取 WavCube 表征

WavCube 表征是一个连续的低帧率语义表征，维度为 128，帧率约 50 Hz
（16kHz 音频下每 320 个采样点对应一帧）。

用法:
    python wav_to_feature.py \
        --audio 19_198_000000_000002.wav \
        --config configs/WavCube-stage2.yaml \
        --ckpt logs/WavCube/checkpoints/vocos_checkpoint_epoch=177_step=195000_val_loss=3.3080.ckpt \
        --output 19_198_000000_000002.pt

输出格式（.pt 文件）:
    {
        "feature":     Tensor [T_feat, 128],  # WavCube 表征，float32
        "sample_rate": int,                   # 原始音频采样率
        "source":      str,                   # 原始音频文件名
    }
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from vocos import Vocos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract WavCube representation from audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="19_198_000000_000002.wav",
        help="输入音频路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/WavCube-stage2.yaml",
        help="模型配置文件路径",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="logs/WavCube/checkpoints/vocos_checkpoint_epoch=177_step=195000_val_loss=3.3080.ckpt",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 .pt 文件路径（默认与音频同名，后缀改为 .pt）",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="目标采样率")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_model(config_path: str, ckpt_path: str, device: str) -> Vocos:
    """加载 WavCube 模型。"""
    print(f"[INFO] Loading config     : {config_path}")
    vocos = Vocos.from_config(config_path)

    print(f"[INFO] Loading checkpoint : {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = vocos.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys  ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"[INFO] Unexpected keys (ignored): {len(unexpected)}")

    vocos = vocos.to(device)
    vocos.eval()
    return vocos


def load_audio(path: str, sample_rate: int, device: str) -> torch.Tensor:
    """加载音频并转换为单通道、指定采样率的 Tensor，shape [1, T]。"""
    audio, sr = torchaudio.load(path)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        print(f"[WARN] Resampling {sr} Hz -> {sample_rate} Hz")
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.to(device)  # [1, T]


def main():
    args = parse_args()

    # 输出路径
    output_path = args.output or str(Path(args.audio).with_suffix(".pt"))

    # 加载模型
    vocos = load_model(args.config, args.ckpt, args.device)

    # 加载音频
    audio = load_audio(args.audio, args.sample_rate, args.device)  # [1, T]
    duration = audio.shape[-1] / args.sample_rate
    print(f"[INFO] Audio  : {args.audio}  |  shape={tuple(audio.shape)}  |  {duration:.2f} s")

    # -------------------------------------------------------------------------
    # 提取 WavCube 表征
    #   feature_extractor.infer() 执行：
    #     WavLM 编码 -> encoder transformer (3 层) -> 线性投影到 latent_dim(128)
    #   返回 z_hat，shape [B, T_feat, 128]，帧率约 50 Hz
    # -------------------------------------------------------------------------
    with torch.no_grad():
        feature = vocos.feature_extractor.infer(audio)  # [B, T_feat, 128]

    feature = feature.squeeze(0).cpu()  # [T_feat, 128]
    token_rate = feature.shape[0] / duration
    print(f"[INFO] Feature: shape={tuple(feature.shape)}  |  token_rate≈{token_rate:.1f} Hz  |  dim={feature.shape[1]}")

    # 保存表征
    payload = {
        "feature": feature,             # Tensor [T_feat, 128], float32
        "sample_rate": args.sample_rate,
        "source": Path(args.audio).name,
    }
    torch.save(payload, output_path)
    print(f"[INFO] Feature saved -> {output_path}")


if __name__ == "__main__":
    main()
