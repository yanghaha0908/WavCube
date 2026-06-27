"""
Example 2: Recover audio from a WavCube representation

Feed the .pt file saved by wav_to_feature.py (or any tensor of shape [T_feat, 128])
into the WavCube Backbone (MiMoBackbone) to reconstruct the original waveform.

Usage:
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
        help="Path to the input .pt file (output of wav_to_feature.py)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/WavCube-stage2.yaml",
        help="Path to the model config file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="logs/WavCube/checkpoints/vocos_checkpoint_epoch=177_step=195000_val_loss=3.3080.ckpt",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output wav file (defaults to *_recon.wav generated alongside the .pt file)",
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


def load_feature(path: str, device: str) -> tuple:
    """
    Read the WavCube representation from a .pt file.

    Returns:
        feature  : Tensor [T_feat, 128]
        sample_rate: int
    """
    payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        feature = payload["feature"]          # [T_feat, 128]
        sample_rate = payload.get("sample_rate", 16000)
        source = payload.get("source", Path(path).name)
    else:
        # Handle the case where a Tensor was saved directly
        feature = payload
        sample_rate = 16000
        source = Path(path).name

    print(f"[INFO] Feature: source={source}  |  shape={tuple(feature.shape)}")
    return feature.to(device), sample_rate


def main():
    args = parse_args()

    # Output path
    if args.output is None:
        pt_path = Path(args.feature)
        output_path = str(pt_path.with_name(pt_path.stem + "_recon.wav"))
    else:
        output_path = args.output

    # Load model
    vocos = load_model(args.config, args.ckpt, args.device)

    # Load representation
    feature, sample_rate = load_feature(args.feature, args.device)  # [T_feat, 128]

    # -------------------------------------------------------------------------
    # Reconstruct audio from the WavCube representation
    #   vocos.decode() runs:
    #     MiMoBackbone (upsample_proj + AudioDecoder.forward_50hz)
    #     turns [B, T_feat, 128] -> waveform [B, T_wav]
    # -------------------------------------------------------------------------
    feature_batch = feature.unsqueeze(0)  # [1, T_feat, 128]

    with torch.no_grad():
        recon = vocos.decode(feature_batch)   # [B, T_wav]  or  [B, 1, T_wav]

    # Normalize the shape to [1, T_wav] (torchaudio.save requires a 2-D tensor)
    recon = recon.squeeze(0)          # [T_wav] or [1, T_wav]
    if recon.dim() == 1:
        recon = recon.unsqueeze(0)    # [1, T_wav]
    recon = recon.cpu()

    duration = recon.shape[-1] / sample_rate
    print(f"[INFO] Reconstructed audio: shape={tuple(recon.shape)}  |  {duration:.2f} s")

    # Save audio
    torchaudio.save(output_path, recon, sample_rate=sample_rate)
    print(f"[INFO] Audio saved -> {output_path}")


if __name__ == "__main__":
    main()
