from typing import Optional, List

import torch
from torch import nn
from torch.nn.utils import weight_norm

from vocos.modules import ConvNeXtBlock, ResBlock1, AdaLayerNorm
from mimo_audio_tokenizer.model import AudioDecoder
from mimo_audio_tokenizer.config import MiMoAudioTokenizerConfig

class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        if x.shape[1] != self.embed.in_channels:
            x = x.transpose(1,2)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x #(B, T, D)


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self, input_channels, dim, num_blocks, layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(nn.Conv1d(input_channels, dim, kernel_size=3, padding=1))
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x


class MiMoBackbone(Backbone):
    def __init__(
        self,
        d_model: int = 1024,
        decoder_attention_heads: int = 16,
        decoder_ffn_dim: int = 5120,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        window_size: int = 640,
        nfft : int = 640,
        upsample: bool = False,
        latent_dim: int = 128,
        decoder_layers: int = 32,
    ):
        super().__init__()
        
        self.config = MiMoAudioTokenizerConfig(
            d_model = d_model,
            decoder_layers = decoder_layers,
            decoder_attention_heads = decoder_attention_heads,
            decoder_ffn_dim = decoder_ffn_dim,
            sampling_rate = sampling_rate,
            hop_length = hop_length,
            window_size = window_size,
            nfft = nfft,
        )
        
        self.decoder = AudioDecoder(self.config)
        
        self.upsample = upsample
        if upsample:
            self.upsample_proj = nn.Conv1d(latent_dim, d_model, kernel_size=1)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        b, l, _ = x.shape
        codes_lens = torch.full((b,), l, device=x.device, dtype=torch.long)
        if self.upsample:
            x = x.transpose(1, 2)
            x = self.upsample_proj(x)
            x = x.transpose(1, 2)
        recon_wav, wav_length = self.decoder.forward_50hz(x, codes_lens)
        recon_wav = recon_wav.squeeze(1)
        return recon_wav
    
    def load_from_ckpt(self, model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu')

        new_state_dict = {}
        
        for k, v in state_dict['state_dict'].items():
            if k.startswith("backbone."):
                new_key = k.replace("backbone.", "")
                new_state_dict[new_key] = v
                
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        
        if len(missing) > 0:
            print(f"[Warning] Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"[Info] Unexpected keys (ignored): {unexpected}")
            
        print("Model loaded successfully.")