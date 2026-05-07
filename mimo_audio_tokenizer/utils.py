# Copyright 2025 Xiaomi Corporation.
from typing import Optional, List
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.nn.utils.rnn import pad_sequence

from mimo_audio_tokenizer.config import MiMoAudioTokenizerConfig

MEL_TRANSFORM = None


def compute_default_rope_parameters(
    config: Optional[dict] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError("Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
                         f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}")
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def rotate_half(x):
    """Rotates half the hidden dims of the input.

    This function is used in rotary position embedding (RoPE) to rotate the input tensor.
    It splits the last dimension in half and applies a rotation by swapping and negating
    the first half.

    Args:
        x (torch.Tensor): Input tensor with shape [..., dim], where dim is even.

    Returns:
        torch.Tensor: Rotated tensor with the same shape as input, where the first half
                     of the last dimension is negated and swapped with the second half.
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The input tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def get_position_ids(lengths):
    """Generate position IDs for packed sequences.

    This function creates position IDs for sequences that have been packed together.
    Each sequence starts with position ID 0 and increments within its own sequence,
    regardless of other sequences in the batch.

    Args:
        lengths (torch.Tensor): Tensor containing the length of each sequence in the batch.
                               Shape: [batch_size]

    Returns:
        torch.Tensor: Position IDs for all packed sequences. Shape: [sum(lengths)]
                     Each sequence starts from 0 and increments up to its length-1.

    Example:
        If lengths = [3, 2], the output will be [0, 1, 2, 0, 1]
    """
    total_len = lengths.sum()
    offset = torch.cat([torch.zeros(1).to(lengths), lengths[:-1].cumsum(dim=0)])
    offset = torch.repeat_interleave(offset, lengths)
    position_ids = torch.arange(0, total_len).to(offset) - offset
    return position_ids


def get_sequence_mask(inputs, inputs_length):
    """Generate sequence mask and unpacking index for batch processing.

    This function creates a mask to identify valid positions in padded sequences
    and generates an unpacking index for converting between packed and unpacked formats.

    Args:
        inputs (torch.Tensor): Input tensor, can be 2D or 3D.
                              If 3D: [batch_size, seq_len, hidden_dim]
                              If 2D: shape is inferred from inputs_length
        inputs_length (torch.Tensor): Actual lengths of sequences in the batch.
                                     Shape: [batch_size]

    Returns:
        tuple: A tuple containing:
            - sequence_mask (torch.Tensor): Boolean mask indicating valid positions.
                                           Shape: [batch_size, max_seq_len, 1]
            - unpacking_index (torch.Tensor): Index tensor for unpacking operations.
                                             Shape: [sum(inputs_length)]
    """
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(bsz, tgt_len, 1)
    unpacking_index = torch.cumsum(sequence_mask.to(torch.int64).view(-1), dim=0) - 1  # 转成下标
    return sequence_mask, unpacking_index


def packing(hidden_states, lengths, sequence_mask=None, unpacking_index=None):
    """
    Pack hidden states from batch format to packed format for efficient processing.

    Args:
        hidden_states: Input hidden states, shape [batch_size, seq_len, hidden_dim]
        lengths: Sequence lengths for each batch item, shape [batch_size]
        sequence_mask: Optional sequence mask, if None will be computed
        unpacking_index: Optional unpacking index, if None will be computed

    Returns:
        Tensor: Packed hidden states, shape [sum(lengths), hidden_dim]
    """
    if sequence_mask is None or unpacking_index is None:
        sequence_mask, unpacking_index = get_sequence_mask(hidden_states, lengths)

    packed_hidden_states = torch.masked_select(hidden_states, sequence_mask).view(torch.sum(lengths),
                                                                                  hidden_states.shape[-1])
    return packed_hidden_states


def unpacking(hidden_states, lengths, sequence_mask=None, unpacking_index=None):
    """Unpack hidden states from packed format back to batch format.

    This function converts packed hidden states (where all valid tokens are
    concatenated) back to the standard batch format with padding.

    Args:
        hidden_states (torch.Tensor): Packed hidden states with shape [sum(lengths), hidden_dim]
        lengths (torch.Tensor): Sequence lengths for each batch item, shape [batch_size]
        sequence_mask (torch.Tensor, optional): Sequence mask for valid positions.
                                               If None, will be computed.
        unpacking_index (torch.Tensor, optional): Index tensor for unpacking.
                                                 If None, will be computed.

    Returns:
        torch.Tensor: Unpacked hidden states in batch format.
                     Shape: [batch_size, max_seq_len, hidden_dim]
                     Invalid positions are filled with zeros.
    """
    bsz = lengths.shape[0]
    if sequence_mask is None or unpacking_index is None:
        sequence_mask, unpacking_index = get_sequence_mask(hidden_states, lengths)
    hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(bsz, torch.max(lengths),
                                                                               hidden_states.shape[-1])
    hidden_states = torch.where(sequence_mask, hidden_states, 0)  # 3d (bsz, max_input_len, d)
    return hidden_states


def load_audio(file: str, sr: int = 24000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A torch.Tensor containing the audio waveform, in float32 dtype.
    """
    audio, sample_rate = torchaudio.load(file)
    if sample_rate != sr:
        audio = torchaudio.transforms.Resample(sample_rate, sr)(audio)
    if audio.ndim == 2:
        audio = audio.mean(dim=0)  # (wav_len,)
    return audio


def padding(data: List[torch.Tensor]):
    """ Padding the data into batch data

    Parameters
    ----------
        data: List[Tensor], shape of Tensor (n_mels, T)

    Returns:
    -------
        feats [B, n_mels, T_max], feats lengths [B]
    """
    sample = data
    assert isinstance(sample, list)
    feats_lengths = torch.tensor([s.size(1) for s in sample], dtype=torch.int32, device=sample[0].device)
    feats = [s.t() for s in sample]
    padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)

    return padded_feats.transpose(1, 2), feats_lengths


def mel_spectrogram(audio, config: MiMoAudioTokenizerConfig):
    """
    Convert raw audio waveform to mel spectrogram representation.

    This function applies mel spectrogram transformation to the input audio signal
    and converts it to log scale for better numerical stability and feature representation.

    Args:
        audio: Input audio waveform tensor, shape [audio_length]

    Returns:
        Tensor: Log mel spectrogram features, shape [n_mels, seq_len]
               where n_mels is the number of mel frequency bins and
               seq_len is the sequence length after STFT transformation
    """
    global MEL_TRANSFORM
    if MEL_TRANSFORM is None:
        MEL_TRANSFORM = MelSpectrogram(
            sample_rate=config.sampling_rate,
            n_fft=config.nfft,
            hop_length=config.hop_length,
            win_length=config.window_size,
            f_min=config.fmin,
            f_max=config.fmax,
            n_mels=config.n_mels,
            power=1.0,
            center=True,
        )
    spec = MEL_TRANSFORM(audio[None, :])  # (1, n_mels, seq_len)
    return torch.log(torch.clip(spec, min=1e-7)).squeeze()  # (n_mels, seq_len)
