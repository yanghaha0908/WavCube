# Copyright 2025 Xiaomi Corporation.
import torch
import torch.nn as nn
import os
import math
import time

from mimo_audio_tokenizer.config import MiMoAudioTokenizerConfig
from mimo_audio_tokenizer.modules.quantizer import ResidualVectorQuantizer
from mimo_audio_tokenizer.modules.layer import (
    CausalConvTranspose1d,
    TransformerLayer,
    RotaryEmbedding,
    TransformerVocos,
)
from mimo_audio_tokenizer.utils import (
    get_position_ids,
    packing,
    unpacking,
)


class AudioEncoder(nn.Module):

    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()

        config._attn_implementation = 'flash_attention_2'
        assert config.activation_function == 'gelu'
        assert config.position_embedding_type == 'rope'
        assert config.avg_pooler != 1
        assert config.num_quantizers != 0

        self.config = config
        self.max_source_positions = (config.max_audio_seconds * config.sampling_rate //
                                     config.hop_length) // config.stride_size
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.skip_layer_idx = config.encoder_skip_layer_id
        self.conv1 = nn.Conv1d(config.n_mels, config.d_model, kernel_size=config.kernel_size, padding=1)
        self.conv2 = nn.Conv1d(config.d_model,
                               config.d_model,
                               kernel_size=config.kernel_size,
                               stride=config.stride_size,
                               padding=1)
        self.position_embedding = RotaryEmbedding(config.rope_theta, config.d_model // config.encoder_attention_heads,
                                                  self.max_source_positions, config.rope_type)

        self.layers = nn.ModuleList([
            TransformerLayer(nn.functional.gelu,
                             config.d_model,
                             config.encoder_attention_heads,
                             config.encoder_ffn_dim,
                             causal=self.config.encoder_causal,
                             ln_type=self.config.ln_type,
                             attn_window_size=self.config.encoder_attn_window_size)
            for _ in range(config.encoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.down_sample_layer = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model, config.avg_pooler, config.avg_pooler, bias=False), nn.GELU())
        self.down_sample_norm = nn.LayerNorm(config.d_model)

        self.quantizer = ResidualVectorQuantizer(dimension=self.config.d_model,
                                                 n_q=self.config.num_quantizers,
                                                 bins=self.config.codebook_size,
                                                 threshold_ema_dead_code=self.config.threshold_ema_dead_code)

        self.adapter = nn.Linear(config.d_model, config.vocoder_dim)
        
    @torch.no_grad()
    def get_features(self, input_features, output_length):
        """
        Extract encoder features from input features

        Args:
            input_features: Input mel spectrogram features, shape [batch_size, n_mels, seq_len]
            output_length: Output length, shape [batch_size]

        Returns:
            Tuple[Tensor, Tensor]:
                - hidden_states: Hidden states, shape [batch_size, max_len, d_model]
                - output_length: Updated output length, shape [batch_size]
        """
        input_features = input_features.to(self.conv1.weight)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))  # (bs, channels, frames)
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))  # (bs, channels, frames // 2)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (bs, frames, channels // 2)
        hidden_states = inputs_embeds

        position_ids = get_position_ids(output_length).long().to(input_features.device)
        rope_position_embeddings = self.position_embedding(input_features, position_ids)

        hidden_states = packing(hidden_states, output_length)
        skip_connect_hidden_states = 0.0
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states,
                                          output_length,
                                          rope_position_embeddings=rope_position_embeddings)
            if (self.skip_layer_idx is not None) and idx == self.skip_layer_idx - 1:
                skip_connect_hidden_states = hidden_states.clone()
        hidden_states += skip_connect_hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = unpacking(hidden_states, output_length)

        if hidden_states.size(1) % self.config.avg_pooler:
            pad_len = self.config.avg_pooler - hidden_states.size(1) % self.config.avg_pooler
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_len), mode='constant', value=0.)
        hidden_states = self.down_sample_layer(hidden_states.transpose(1, 2))
        output_length = output_length // self.config.avg_pooler + (output_length % self.config.avg_pooler != 0).int()
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.down_sample_norm(hidden_states)

        return hidden_states, output_length

    @torch.no_grad()
    def get_features_50hz(self, input_features, input_lens):
        """
        Identical logic to get_features, but stops after the Transformer layers, skipping the final down_sample module.
        """
        output_length = self.get_output_length(input_lens)
            
        input_features = input_features.to(self.conv1.weight)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))  # (bs, channels, frames)
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))  # (bs, channels, frames // 2)
        inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (bs, frames, channels // 2)
        hidden_states = inputs_embeds

        position_ids = get_position_ids(output_length).long().to(input_features.device)
        rope_position_embeddings = self.position_embedding(input_features, position_ids)

        hidden_states = packing(hidden_states, output_length)
        skip_connect_hidden_states = 0.0
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states,
                                          output_length,
                                          rope_position_embeddings=rope_position_embeddings)
            if (self.skip_layer_idx is not None) and idx == self.skip_layer_idx - 1:
                skip_connect_hidden_states = hidden_states.clone()
        hidden_states += skip_connect_hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = unpacking(hidden_states, output_length)

        # The avg_pooler padding and down_sample_layer convolution logic has been removed here
        # Only the unpacked hidden_states and the corresponding output_length are kept (now determined by conv2)
        hidden_states = self.adapter(hidden_states) #B,T,D
        
        return hidden_states, output_length #B,T,D
    
    def get_output_length(self, mel_len):
        """
        Calculate output length for a given mel spectrogram length

        Args:
            mel_len: Mel spectrogram length, shape [batch_size] or a single integer

        Returns:
            Tensor or int: Calculated output length, with the same shape as input
        """
        tgt_len = mel_len + 3 - self.config.kernel_size
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1

    @torch.no_grad()
    def encode(
        self,
        input_features,
        input_lens=None,
        output_length=None,
        n_q=None,
    ):
        """
        Encode input features into discrete codes

        Args:
            input_features: Input mel spectrogram features, shape [batch_size, n_mels, seq_len]
            input_lens: Input lengths, shape [batch_size]
            output_length: Optional, output lengths, shape [batch_size], if None calculated from input_lens
            n_q: Optional, number of quantizers, if None uses all quantizers

        Returns:
            Tuple[List[Tensor], Tensor]:
                - total_codes: List of quantized codes, each with shape [batch_size, max_len, num_quantizers]
                - output_length: Output lengths, shape [batch_size]
        """
        if output_length is None:
            output_length = self.get_output_length(input_lens)

        hidden_states, output_length = self.get_features(
            input_features=input_features, output_length=output_length)  # (batch_size, max_len, d_model), (batch_size)

        hidden_states = packing(hidden_states, output_length)  # (packed_len, d_model)
        self.quantizer.float()
        codes = self.quantizer.encode(hidden_states.float(), n_q=n_q)  # (n_q, packed_len)
        codes = codes.transpose(0, 1)  # (packed_len, n_q)
        codes = unpacking(codes, output_length)  # (batch_size, max_len, n_q)

        return codes, output_length

    @torch.no_grad()
    def decode_vq(self, codes, output_length):
        """
        Decode quantized codes into hidden states

        Args:
            codes: List of quantized codes, each with shape [batch_size, max_len, num_quantizers]

        Returns:
            Tensor: Decoded hidden states, shape [batch_size, max_len, d_model]
        """
        codes = packing(codes, output_length)  # (packed_len, n_q)
        codes = codes.transpose(0, 1)  # (n_q, packed_len)
        self.quantizer.float()
        hidden_states = self.quantizer.decode(codes)  # (packed_len, d_model)
        hidden_states = unpacking(hidden_states, output_length)  # (batch_size, max_len, d_model)
        return hidden_states

    @classmethod
    def load_from_pretrained(cls, config, model_path, process=True):
        """
        Load encoder from pretrained model

        Args:
            config: Model configuration
            model_path: Pretrained model path
            process: Whether to process parameter names, defaults to True

        Returns:
            AudioEncoder: Encoder model loaded with pretrained weights
        """
        model = cls(config)
        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            params_total = load_file(model_path)
        else:
            params_total = torch.load(model_path, map_location='cpu')
        params = {}
        for k in params_total.keys():
            if process and 'audio_tokenizer.encoder' in k:
                params[k.replace('audio_tokenizer.encoder.', '')] = params_total[k]
            elif 'encoder' in k:
                params[k.replace('encoder.', '')] = params_total[k]
        model.load_state_dict(params)
        return model

    @classmethod
    def from_pretrained(cls, model_path, process=True):
        config = MiMoAudioTokenizerConfig.from_pretrained(model_path)
        if os.path.isfile(f'{model_path}/model.safetensors'):
            return cls.load_from_pretrained(config, f'{model_path}/model.safetensors', process)
        else:
            return cls.load_from_pretrained(config, f'{model_path}/pytorch_model.bin', process)


class AudioDecoder(nn.Module):

    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()
        assert config.position_embedding_type == 'rope'
        assert config.avg_pooler != 1

        self.config = config
        self.max_source_positions = self.config.max_audio_seconds * self.config.sampling_rate // self.config.hop_length

        self.dconv1 = CausalConvTranspose1d(
            self.config.d_model,
            self.config.d_model,
            self.config.avg_pooler,
            self.config.avg_pooler,
        )
        self.position_embedding = RotaryEmbedding(config.rope_theta, config.d_model // config.decoder_attention_heads,
                                                  self.max_source_positions, config.rope_type)
        # causal transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                nn.functional.gelu,
                self.config.d_model,
                self.config.decoder_attention_heads,
                self.config.decoder_ffn_dim,
                causal=self.config.decoder_causal,  # causal
                ln_type=self.config.ln_type,
                attn_window_size=self.config.decoder_attn_window_size,
            ) for _ in range(self.config.decoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(self.config.d_model)
        self.dconv2 = CausalConvTranspose1d(self.config.d_model, self.config.n_mels, self.config.decoder_kernel_size,
                                            self.config.decoder_stride_size)
        self.vocoder = TransformerVocos(config)

    def forward(
        self,
        audio_embed,
        input_length,
    ):
        """
        Forward pass of the decoder

        Args:
            audio_embed: Audio embeddings, shape [batch_size, seq_len, d_model]
            input_length: Input lengths, shape [batch_size]

        Returns:
            Tensor: Reconstructed waveform, shape [batch_size, 1, wav_length]
            Tensor: Output lengths, shape [batch_size]
        """
        assert (audio_embed.shape[-1] == self.config.d_model) # B,T,D
        audio_embed = audio_embed.to(self.layer_norm.weight)  # device and type
        audio_embed, output_length = self.dconv1(audio_embed, input_length, output_dim=3)  # (b, l*2, d_model)
        hidden_states = audio_embed

        position_ids = get_position_ids(output_length).long().to(hidden_states.device)
        rope_position_embeddings = self.position_embedding(hidden_states, position_ids)

        hidden_states = packing(hidden_states, output_length)
        for _, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states,
                                          output_length,
                                          rope_position_embeddings=rope_position_embeddings)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = unpacking(hidden_states, output_length)

        coarse_mel, output_length = self.dconv2(hidden_states, output_length, output_dim=3)

        recon_wav, wav_length = self.vocoder(
            x=coarse_mel,  # (B, L, C)
            input_length=output_length)  # (B, 1, L_wav), (B)

        return recon_wav, wav_length

    def forward_50hz(
        self,
        audio_embed,
        input_length,
    ):
        """
        Forward pass of the decoder for 50Hz features (Skipping the first upsampling layer dconv1)

        Args:
            audio_embed: Audio embeddings, shape [batch_size, seq_len, d_model] 
                        (Now seq_len corresponds to 50Hz resolution)
            input_length: Input lengths, shape [batch_size]

        Returns:
            Tensor: Reconstructed waveform, shape [batch_size, 1, wav_length]
            Tensor: Output lengths, shape [batch_size]
        """
        assert (audio_embed.shape[-1] == self.config.d_model)
        audio_embed = audio_embed.to(self.layer_norm.weight)
        # ==========================================================
        # Modification: skip self.dconv1 (the first upsampling layer)
        # Original logic: audio_embed, output_length = self.dconv1(audio_embed, input_length, output_dim=3)
        # New logic: pass through directly.
        #         Since the input is already 50Hz and the Transformer itself operates at 50Hz resolution,
        #         we do not need dconv1 to upsample from 25Hz to 50Hz.
        # ==========================================================
        hidden_states = audio_embed
        output_length = input_length  # length stays unchanged (50Hz)

        position_ids = get_position_ids(output_length).long().to(hidden_states.device)
        rope_position_embeddings = self.position_embedding(hidden_states, position_ids)

        hidden_states = packing(hidden_states, output_length)
        for _, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states,
                                          output_length,
                                          rope_position_embeddings=rope_position_embeddings)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = unpacking(hidden_states, output_length)

        coarse_mel, output_length = self.dconv2(hidden_states, output_length, output_dim=3)

        recon_wav, wav_length = self.vocoder(
            x=coarse_mel,  # (B, L, C)
            input_length=output_length)  # (B, 1, L_wav), (B)

        return recon_wav, wav_length

    @classmethod
    def load_from_pretrained(cls, config, model_path, process=True):
        """
        Load decoder from pretrained model

        Args:
            config: Model configuration
            model_path: Pretrained model path
            process: Whether to process parameter names, defaults to True

        Returns:
            AudioDecoder: Decoder model loaded with pretrained weights
        """
        model = cls(config)
        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            params_total = load_file(model_path)
        else:
            params_total = torch.load(model_path, map_location='cpu')
        params = {}
        for k in params_total.keys():
            if process and 'audio_tokenizer.decoder' in k:
                params[k.replace('audio_tokenizer.decoder.', '')] = params_total[k]
            elif 'decoder' in k:
                params[k.replace('decoder.', '')] = params_total[k]
        model.load_state_dict(params)
        return model

    @classmethod
    def from_pretrained(cls, model_path, process=True):
        """
        Load decoder from pretrained model path

        Args:
            model_path: Directory path containing the pretrained model
            process: Whether to process parameter names, defaults to True

        Returns:
            AudioDecoder: Decoder model loaded with pretrained weights
        """
        config = MiMoAudioTokenizerConfig.from_pretrained(model_path)
        if os.path.isfile(f'{model_path}/model.safetensors'):
            return cls.load_from_pretrained(config, f'{model_path}/model.safetensors', process)
        else:
            return cls.load_from_pretrained(config, f'{model_path}/pytorch_model.bin', process)


class MiMoAudioTokenizer(nn.Module):

    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config=config)
        self.decoder = AudioDecoder(config=config)

    @torch.no_grad()
    def encode(self, mels, mels_lens, n_q=None, **kwargs):
        """
        Encode mel spectrograms into discrete codes

        Args:
            mels: Mel spectrogram features, shape [batch_size, n_mels, seq_len]
            mels_lens: Input lengths, shape [batch_size]
            n_q: Number of quantizers, if None uses all quantizers

        Returns:
            Tuple[List[Tensor], Tensor]:
                - codes: List of quantized codes, each with shape [batch_size, max_len, num_quantizers]
                - encoder_output_length: Output lengths, shape [batch_size]
            Dict: Timing statistics
                - batch_size: Batch size
                - encode_time: Encode time
        """
        timing_stats = {"batch_size": mels.shape[0]}
        start_time = time.time()
        codes, encoder_output_length = self.encoder.encode(
            mels,
            input_lens=mels_lens,
            n_q=n_q,
        )
        end_time = time.time()
        timing_stats["encode_time"] = end_time - start_time
        return codes, encoder_output_length, timing_stats

    @torch.no_grad()
    def decode(self, codes, codes_lens, **kwargs):
        """
        Decode quantized codes into waveform

        Args:
            codes: List of quantized codes, each with shape [batch_size, max_len, num_quantizers]
            output_lens: Output lengths, shape [batch_size]

        Returns:
            Tensor: Reconstructed waveform, shape [batch_size, 1, wav_length]
            Tensor: Output lengths, shape [batch_size]
            Dict: Timing statistics
                - batch_size: Batch size
                - decode_time: Decode time
        """
        timing_stats = {"batch_size": codes.shape[0]}
        start_time = time.time()
        hidden_states = self.encoder.decode_vq(codes, codes_lens)  # (batch_size, max_len, d_model)
        recon_wav, wav_length = self.decoder(hidden_states, codes_lens)
        end_time = time.time()
        timing_stats["decode_time"] = end_time - start_time
        return recon_wav, wav_length, timing_stats

    @classmethod
    def load_from_pretrained(cls, config, model_path, load_encoder_only=False, process=True):
        """
        Load tokenizer from pretrained model

        Args:
            config: Model configuration
            model_path: Pretrained model path
            load_encoder_only: Whether to load only the encoder part, defaults to False
            process: Whether to process parameter names, defaults to True

        Returns:
            MiMoAudioTokenizer: Tokenizer model loaded with pretrained weights
        """
        model = cls(config)
        from safetensors.torch import load_file
        params_total = load_file(model_path)
        if process:
            params = {}
            for k in params_total.keys():
                if 'audio_tokenizer' in k:
                    if load_encoder_only:
                        if 'encoder' not in k:
                            continue
                        else:
                            params[k.replace('audio_tokenizer.encoder.', '')] = params_total[k]
                    else:
                        params[k.replace('audio_tokenizer.', '')] = params_total[k]
        else:
            params = params_total
        if load_encoder_only:
            model.encoder.load_state_dict(params)
        else:
            model.load_state_dict(params)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, process=True, verbose=False):
        """
        Load tokenizer from pretrained model path

        Args:
            pretrained_model_name_or_path: Directory path containing the pretrained model
            process: Whether to process parameter names, defaults to True

        Returns:
            MiMoAudioTokenizer: Tokenizer model loaded with pretrained weights
        """
        config = MiMoAudioTokenizerConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        if os.path.isfile(f'{pretrained_model_name_or_path}/model.safetensors'):
            model_path = f'{pretrained_model_name_or_path}/model.safetensors'
            from safetensors.torch import load_file
            params_total = load_file(model_path)
        else:
            model_path = f'{pretrained_model_name_or_path}/pytorch_model.bin'
            params_total = torch.load(model_path, map_location='cpu')
        if process:
            params = {}
            for k in params_total.keys():
                if 'audio_tokenizer' in k:
                    params[k.replace('audio_tokenizer.', '')] = params_total[k]
        else:
            params = params_total

        missing_keys, unexpected_keys = model.load_state_dict(params, strict=False)
        if len(missing_keys) > 0 and verbose:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0 and verbose:
            print(f"Unexpected keys: {unexpected_keys}")
        return model

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device
