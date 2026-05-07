# Copyright 2025 Xiaomi Corporation.
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func

from mimo_audio_tokenizer.config import MiMoAudioTokenizerConfig
from mimo_audio_tokenizer.utils import (
    compute_default_rope_parameters,
    apply_rotary_pos_emb,
    get_sequence_mask,
    packing,
    unpacking,
)


class CausalConvTranspose1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, hidden_states, input_length, output_dim=None):
        """
        Forward pass for causal transposed convolution with proper padding.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (N, L, C) for 3D or (N*L, C) for 2D,
                                        where N is batch size, L is sequence length, C is input channels
            input_length (torch.Tensor): Length of each sequence in the batch, shape (N,)
            output_dim (int, optional): Target output dimension. If None, uses input tensor dimension

        Returns:
            tuple: (hidden_states, output_length) where:
                - hidden_states: Output tensor with same dimensionality as input but upsampled length
                                If output_dim <= 2: shape (N*output_L, out_channels)
                                If output_dim > 2: shape (N, output_L, out_channels)
                - output_length: Length of output sequences, shape (N,)
        """
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        bsz = input_length.shape[0]

        if output_dim is None:
            output_dim = hidden_states.dim()
        if hidden_states.dim() <= 2:  # unpack sequence to 3d
            hidden_states = unpacking(hidden_states, input_length)

        hidden_states = hidden_states.transpose(2, 1)  # (N, L, C) -> (N, C, L)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(2, 1)  # (N, C, L) -> (N, L, C)

        causal_padding_right = max(0, kernel_size - stride)
        hidden_states = hidden_states[:, :hidden_states.shape[1] - causal_padding_right, :]
        output_length = (input_length - 1) * stride + kernel_size - causal_padding_right
        sequence_mask, _ = get_sequence_mask(hidden_states, output_length)
        if output_dim <= 2:
            hidden_states = torch.masked_select(hidden_states, sequence_mask).view(-1, self.out_channels)
        else:
            hidden_states = torch.where(sequence_mask, hidden_states, 0)
            hidden_states = hidden_states[:, :torch.max(output_length), :]
        return hidden_states, output_length


class RotaryEmbedding(nn.Module):

    def __init__(self, base, dim, max_seq_len, rope_type="default", device=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.rope_type = rope_type
        assert self.rope_type == "default"
        self.rope_init_fn = compute_default_rope_parameters
        inv_freq, self.attention_scaling = self.rope_init_fn(device=device, base=base, dim=dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Compute rotary position embeddings (RoPE) for the given positions.

        Args:
            x (torch.Tensor): Input tensor used for device and dtype reference, shape can be any
            position_ids (torch.Tensor): Position indices for each token, shape (seq_len,)

        Returns:
            tuple: (cos, sin) where:
                - cos: Cosine values for rotary embeddings, shape (seq_len, dim)
                - sin: Sine values for rotary embeddings, shape (seq_len, dim)
                Both tensors have the same dtype as input x and are scaled by attention_scaling
        """
        inv_freq_expanded = self.inv_freq[:, None].float().expand(-1, 1).to(x.device)
        position_ids_expanded = position_ids[None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):

    def __init__(self, embed_dim, num_heads, window_size=(-1, -1), causal=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.causal = causal

    def forward(self, hidden_states: torch.Tensor, seq_len: torch.Tensor, rope_position_embeddings=None):
        """
        Forward pass for multi-head attention with optional rotary position embeddings.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (total_seq_len, embed_dim),
                                        where total_seq_len is the sum of all sequence lengths in batch
            seq_len (torch.Tensor): Length of each sequence in the batch, shape (batch_size,)
            rope_position_embeddings (tuple, optional): (cos, sin) tensors for rotary position embeddings,
                                                       each of shape (seq_len, head_dim)

        Returns:
            torch.Tensor: Attention output of shape (total_seq_len, embed_dim)
        """
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)

        if rope_position_embeddings:
            cos, sin = rope_position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.int32)
        max_seqlen = torch.max(seq_len).to(torch.int32).detach()
        attn_output = flash_attn_varlen_func(#query_states,
                                            #  key_states,
                                            #  value_states,
                                             query_states.to(torch.float16),  # <--- 强制转 fp16
                                             key_states.to(torch.float16),    # <--- 强制转 fp16
                                             value_states.to(torch.float16),
                                             cu_len,
                                             cu_len,
                                             max_seqlen,
                                             max_seqlen,
                                             causal=self.causal,
                                             window_size=self.window_size)  # (bsz * qlen, nheads, headdim)
        attn_output = attn_output.to(torch.float32) # <--- 
        attn_output = attn_output.reshape(bsz, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class TransformerLayer(nn.Module):

    def __init__(
            self,
            act,
            d_model,
            encoder_attention_heads,
            encoder_ffn_dim,
            causal,
            ln_type="LayerNorm",
            attn_window_size=(-1, -1),
    ):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = Attention(self.embed_dim, encoder_attention_heads, attn_window_size, causal)
        assert ln_type == "LayerNorm"

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.activation_fn = act
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, seq_len: torch.Tensor,
                rope_position_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a transformer layer with self-attention and feedforward network.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (total_seq_len, d_model),
                                        where total_seq_len is the sum of all sequence lengths in batch
            seq_len (torch.Tensor): Length of each sequence in the batch, shape (batch_size,)
            rope_position_embeddings (torch.Tensor): Rotary position embeddings (cos, sin) tuple
                                                    for the attention mechanism

        Returns:
            torch.Tensor: Output tensor of shape (total_seq_len, d_model) after applying
                         self-attention and feedforward transformations with residual connections
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, seq_len, rope_position_embeddings=rope_position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if (hidden_states.dtype == torch.float16 or hidden_states.dtype
                == torch.bfloat16) and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        original_dtype = x.dtype
        S = mag.float() * (x.float() + 1j * y.float())
        audio = self.istft(S)
        audio = audio.to(original_dtype)
        return audio


class TransformerVocos(nn.Module):

    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()
        assert config.activation_function == 'gelu'
        assert config.ln_type == "LayerNorm"

        self.config = config
        self.max_source_positions = self.config.max_audio_seconds * self.config.sampling_rate // self.config.hop_length
        self.embeddings = nn.Linear(config.n_mels, config.vocoder_dim, bias=False)

        self.position_embedding = RotaryEmbedding(config.rope_theta,
                                                  config.vocoder_dim // config.vocoder_attention_heads,
                                                  self.max_source_positions, self.config.rope_type)

        self.layers = nn.ModuleList([
            TransformerLayer(nn.functional.gelu,
                             self.config.vocoder_dim,
                             self.config.vocoder_attention_heads,
                             self.config.vocoder_intermediate_dim,
                             causal=False,
                             ln_type=self.config.ln_type,
                             attn_window_size=self.config.vocoder_attn_window_size)
            for _ in range(self.config.vocoder_num_layers)
        ])

        self.layer_norm = nn.LayerNorm(self.config.vocoder_dim)
        self.hop_size = self.config.hop_length
        self.head = ISTFTHead(self.config.vocoder_dim, self.config.nfft, self.config.hop_length,
                              self.config.vocoder_padding)

    def forward(self, x: torch.Tensor, input_length):
        """
        Forward pass for the Transformer-based vocoder (Vocos).

        Args:
            x (torch.Tensor): Input mel-spectrogram tensor of shape (batch_size, seq_len, n_mels)
            input_length (torch.Tensor): Length of each sequence in the batch, shape (batch_size,)
                                       representing the number of time frames for each sample

        Returns:
            tuple: (audio_output, output_length) where:
                - audio_output: Generated audio waveform of shape (batch_size, 1, audio_length)
                - output_length: Length of output audio sequences, shape (batch_size,)
                               where audio_length = input_length * hop_size
        """
        x = packing(x, input_length)
        x = self.embeddings(x)
        position_ids = torch.arange(0, x.size(0), device=x.device, dtype=torch.long)
        rope_position_embeddings = self.position_embedding(x, position_ids)
        for _, layer in enumerate(self.layers):
            x = layer(x, input_length, rope_position_embeddings=rope_position_embeddings)
        x = self.layer_norm(x)
        x = unpacking(x, input_length)
        x = self.head(x)
        output_length = input_length * self.hop_size
        return x[:, None, :], output_length
