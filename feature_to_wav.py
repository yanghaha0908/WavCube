"""
示例二：从 WavCube 表征恢复音频

将 wav_to_feature.py 保存的 .pt 文件（或任意形状 [T_feat, 128] 的张量）
传入 WavCube Backbone（MiMoBackbone），重建原始波形。

用法:
    python feature_to_wav.py \
        --feature 19_198_000000_000002.pt \
        --config configs/WavCube-stage2.yaml \
        --ckpt logs/WavCube/checkpoints/vocos_checkpoint_epoch=177_step=195000_val_loss=3.3080.ckpt \
        --output 19_198_000000_000002_recon.wav
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from vocos import Vocos


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct audio from WavCube representation."
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="19_198_000000_000002.pt",
        help="输入 .pt 文件路径（wav_to_feature.py 的输出）",
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
        help="输出 wav 文件路径（默认在 .pt 文件同目录下生成 *_recon.wav）",
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


def load_feature(path: str, device: str) -> tuple:
    """
    从 .pt 文件读取 WavCube 表征。

    返回:
        feature  : Tensor [T_feat, 128]
        sample_rate: int
    """
    payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        feature = payload["feature"]          # [T_feat, 128]
        sample_rate = payload.get("sample_rate", 16000)
        source = payload.get("source", Path(path).name)
    else:
        # 兼容直接保存 Tensor 的情况
        feature = payload
        sample_rate = 16000
        source = Path(path).name

    print(f"[INFO] Feature: source={source}  |  shape={tuple(feature.shape)}")
    return feature.to(device), sample_rate


def main():
    args = parse_args()

    # 输出路径
    if args.output is None:
        pt_path = Path(args.feature)
        output_path = str(pt_path.with_name(pt_path.stem + "_recon.wav"))
    else:
        output_path = args.output

    # 加载模型
    vocos = load_model(args.config, args.ckpt, args.device)

    # 加载表征
    feature, sample_rate = load_feature(args.feature, args.device)  # [T_feat, 128]

    # -------------------------------------------------------------------------
    # 从 WavCube 表征重建音频
    #   vocos.decode() 执行：
    #     MiMoBackbone（upsample_proj + AudioDecoder.forward_50hz）
    #     将 [B, T_feat, 128] -> 波形 [B, T_wav]
    # -------------------------------------------------------------------------
    feature_batch = feature.unsqueeze(0)  # [1, T_feat, 128]

    with torch.no_grad():
        recon = vocos.decode(feature_batch)   # [B, T_wav]  or  [B, 1, T_wav]

    # 统一形状为 [1, T_wav]（torchaudio.save 需要二维张量）
    recon = recon.squeeze(0)          # [T_wav] or [1, T_wav]
    if recon.dim() == 1:
        recon = recon.unsqueeze(0)    # [1, T_wav]
    recon = recon.cpu()

    duration = recon.shape[-1] / sample_rate
    print(f"[INFO] Reconstructed audio: shape={tuple(recon.shape)}  |  {duration:.2f} s")

    # 保存音频
    torchaudio.save(output_path, recon, sample_rate=sample_rate)
    print(f"[INFO] Audio saved -> {output_path}")


if __name__ == "__main__":
    main()
