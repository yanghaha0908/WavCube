"""
Example 1: Extract WavCube representation from audio

The WavCube representation is a continuous low-frame-rate semantic representation
with dimension 128 and a frame rate of about 50 Hz (one frame per 320 samples for
16kHz audio).

Usage:
    python wav_to_feature.py \
        --audio 19_198_000000_000002.wav \
        --config configs/WavCube-stage2.yaml \
        --ckpt logs/WavCube/checkpoints/vocos_checkpoint_epoch=177_step=195000_val_loss=3.3080.ckpt \
        --output 19_198_000000_000002.pt

Output format (.pt file):
    {
        "feature":     Tensor [T_feat, 128],  # WavCube representation, float32
        "sample_rate": int,                   # original audio sample rate
        "source":      str,                   # original audio file name
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
        help="Input audio path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/WavCube-stage2.yaml",
        help="Model config file path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="logs/WavCube/checkpoints/vocos_checkpoint_epoch=177_step=195000_val_loss=3.3080.ckpt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt file path (defaults to the audio's name with the suffix changed to .pt)",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_model(config_path: str, ckpt_path: str, device: str) -> Vocos:
    """Load the WavCube model."""
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
    """Load audio and convert it to a single-channel Tensor at the given sample rate, shape [1, T]."""
    audio, sr = torchaudio.load(path)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        print(f"[WARN] Resampling {sr} Hz -> {sample_rate} Hz")
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.to(device)  # [1, T]


def main():
    args = parse_args()

    # Output path
    output_path = args.output or str(Path(args.audio).with_suffix(".pt"))

    # Load model
    vocos = load_model(args.config, args.ckpt, args.device)

    # Load audio
    audio = load_audio(args.audio, args.sample_rate, args.device)  # [1, T]
    duration = audio.shape[-1] / args.sample_rate
    print(f"[INFO] Audio  : {args.audio}  |  shape={tuple(audio.shape)}  |  {duration:.2f} s")

    # -------------------------------------------------------------------------
    # Extract WavCube representation
    #   feature_extractor.infer() runs:
    #     WavLM encoding -> encoder transformer (3 layers) -> linear projection to latent_dim(128)
    #   returns z_hat, shape [B, T_feat, 128], frame rate about 50 Hz
    # -------------------------------------------------------------------------
    with torch.no_grad():
        feature = vocos.feature_extractor.infer(audio)  # [B, T_feat, 128]

    feature = feature.squeeze(0).cpu()  # [T_feat, 128]
    token_rate = feature.shape[0] / duration
    print(f"[INFO] Feature: shape={tuple(feature.shape)}  |  token_rate≈{token_rate:.1f} Hz  |  dim={feature.shape[1]}")

    # Save representation
    payload = {
        "feature": feature,             # Tensor [T_feat, 128], float32
        "sample_rate": args.sample_rate,
        "source": Path(args.audio).name,
    }
    torch.save(payload, output_path)
    print(f"[INFO] Feature saved -> {output_path}")


if __name__ == "__main__":
    main()
