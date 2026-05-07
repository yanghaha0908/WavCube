# Copyright 2025 Xiaomi Corporation.
import json
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List


@dataclass
class MiMoAudioTokenizerConfig:
    max_audio_seconds: int = field(default=1800)
    stride_size: int = field(default=2)
    avg_pooler: int = field(default=2)
    d_model: int = field(default=1280)
    scale_embedding: bool = field(default=False)
    kernel_size: int = field(default=3)
    activation_function: str = field(default='gelu')
    encoder_layers: int = field(default=32)
    encoder_skip_layer_id: int = field(default=3)
    encoder_attention_heads: int = field(default=20)
    encoder_ffn_dim: int = field(default=5120)
    encoder_causal: bool = field(default=False)
    encoder_attn_window_size: List[int] = field(default_factory=lambda: (-1, -1))
    decoder_layers: int = field(default=32)
    decoder_attention_heads: int = field(default=20)
    decoder_ffn_dim: int = field(default=5120)
    decoder_kernel_size: int = field(default=3)
    decoder_stride_size: int = field(default=2)
    decoder_causal: bool = field(default=True)
    decoder_attn_window_size: List[int] = field(default_factory=lambda: (-1, -1))
    nfft: int = field(default=960)
    vocoder_dim: int = field(default=256)
    vocoder_intermediate_dim: int = field(default=1024)
    vocoder_num_layers: int = field(default=16)
    n_mels: int = field(default=128)
    sampling_rate: int = field(default=24000)
    hop_length: int = field(default=240)
    window_size: int = field(default=960)
    vocoder_padding: str = field(default='same')
    fmin: int = field(default=0)
    fmax: Optional[int] = field(default=None)
    num_quantizers: int = field(default=20)
    codebook_size: List[int] = field(
        default_factory=lambda:
        [1024, 1024, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128])
    threshold_ema_dead_code: int = field(default=2)
    position_embedding_type: str = field(default='rope')
    rope_theta: int = field(default=10000)
    rope_type: str = field(default='default')
    ln_type: str = field(default="LayerNorm")
    vocoder_attention_heads: int = field(default=16)
    vocoder_attn_window_size: List[int] = field(default_factory=lambda: (40, 10))

    def save_pretrained(self, save_directory: str):
        """
        Save the configuration to a directory as config.json.

        Args:
            save_directory: Directory to save the config file to.
        """
        # Convert dataclass to dict
        config_dict = asdict(self)
        # Write to JSON file
        config_path = "{}/config.json".format(save_directory)
        with open(config_path, 'w+', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """
        Load configuration from a directory containing config.json.

        Args:
            pretrained_model_name_or_path: Path to directory containing config.json or path to config.json file.

        Returns:
            MiMoAudioTokenizerConfig instance with loaded configuration.
        """
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = "{}/config.json".format(pretrained_model_name_or_path)
        else:
            config_path = pretrained_model_name_or_path

        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found at {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # Filter out any extra fields that might be in the JSON but not in the dataclass
        field_names = {f.name for f in fields(cls)}
        filtered_config = {k: v for k, v in config_dict.items() if k in field_names}

        return cls(**filtered_config)
