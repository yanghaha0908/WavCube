from typing import List

import torch
import torchaudio
from encodec import EncodecModel
from torch import nn
import torch.nn.functional as F

from vocos.modules import safe_log
from transformers import AutoModel, AutoFeatureExtractor
from transformers import WhisperFeatureExtractor
from mimo_audio_tokenizer.model import AudioEncoder
from mimo_audio_tokenizer.config import MiMoAudioTokenizerConfig
import mimo_audio_tokenizer
import copy
from transformers.models.wavlm.modeling_wavlm import _compute_mask_indices

class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
        train_codebooks: bool = False,
    ):
        super().__init__()
        if encodec_model == "encodec_24khz":
            encodec = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            encodec = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        self.encodec = encodec(pretrained=True)
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=max(bandwidths)
        )
        codebook_weights = torch.cat([vq.codebook for vq in self.encodec.quantizer.vq.layers[: self.num_q]], dim=0)
        self.codebook_weights = torch.nn.Parameter(codebook_weights, requires_grad=train_codebooks)
        self.bandwidths = bandwidths

    @torch.no_grad()
    def get_encodec_codes(self, audio):
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)
        return codes

    def forward(self, audio: torch.Tensor, **kwargs):
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")
        self.encodec.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])
        codes = self.get_encodec_codes(audio)
        # Instead of summing in the loop, it stores subsequent VQ dictionaries in a single `self.codebook_weights`
        # with offsets given by the number of bins, and finally summed in a vectorized operation.
        offsets = torch.arange(
            0, self.encodec.quantizer.bins * len(codes), self.encodec.quantizer.bins, device=audio.device
        )
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.codebook_weights).sum(dim=0)
        return features.transpose(1, 2)

class WavLMFeatures(FeatureExtractor):
    def __init__(
        self,
        model_id: str = "microsoft/wavlm-large",
        layer_idx: int = -1, # last_hidden_state
        freeze_model: bool = True,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.layer_idx = layer_idx
        self.freeze_model = freeze_model
        
        if self.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, audio, **kwargs):
        if self.freeze_model:
            self.model.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        
        inputs = self.processor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
        outputs = self.model(input_values=inputs.input_values.squeeze(0).to(audio.device), output_hidden_states=True)

        hidden_states = outputs.hidden_states
        idx = self.layer_idx if self.layer_idx < 0 else min(self.layer_idx, len(hidden_states) - 1)
        features = hidden_states[idx] # B,L,C
    
        return features
    

class WavLMVAEFeatures(FeatureExtractor):
    def __init__(
        self,
        model_id: str = "ckpts/wavlm-large",
        layer_idx: int = -1,
        freeze_model: bool = True,
        latent_dim: int = 128,
        stage: int = 1,
        stage1_ckpt_path: str = None,
        use_vae: bool = False,
        use_sigma_vae: bool = False,
        use_temporal_downsampling: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.layer_idx = layer_idx
        self.freeze_model = freeze_model
        self.stage = stage
        self.stage1_ckpt_path = stage1_ckpt_path
        self.use_vae = use_vae
        self.use_sigma_vae = use_sigma_vae
        self.sigma_scale = 0.5
        self.use_temporal_downsampling = use_temporal_downsampling
        
        self.ssl_dim = self.model.config.hidden_size
        self.latent_dim = latent_dim
        projection_dim = (self.ssl_dim+self.latent_dim)//2
        
        if self.freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
        
        if self.stage==2:
            self.rep_model = AutoModel.from_pretrained(model_id)
                
        original_layers = self.model.encoder.layers
        self.enc_transformer = nn.ModuleList([copy.deepcopy(original_layers[i]) for i in range(3)])
        
        self.enc_projection = nn.Sequential(
            nn.Linear(self.ssl_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, latent_dim),
            nn.LayerNorm(latent_dim, eps=1e-6)
        )      
        
        if self.use_vae:
            self.fc_mu = nn.Linear(latent_dim, latent_dim)
            self.fc_var = nn.Linear(latent_dim, latent_dim)
        
        self.dec_projection = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, self.ssl_dim),
            nn.LayerNorm(self.ssl_dim, eps=1e-6)
        )

        self.dec_transformer = nn.ModuleList([copy.deepcopy(original_layers[i]) for i in range(3)])

        for param in self.enc_transformer.parameters(): 
            param.requires_grad = True
        for param in self.dec_transformer.parameters():
            param.requires_grad = True
        
        self.post_wavlm_layer_norm = nn.LayerNorm(self.ssl_dim, eps = self.model.config.layer_norm_eps)
        self.num_hidden_layers = self.model.config.num_hidden_layers
        
    def forward(self, audio, **kwargs):
        output_attentions=False
        if self.freeze_model:
            self.model.eval()
        if self.stage == 2:
            self.rep_model.eval()
         
        device = audio.device
        
        inputs = self.processor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
        
        if self.stage ==1:
            with torch.no_grad():
                outputs = self.model(input_values=inputs.input_values.squeeze(0).to(device), output_hidden_states=True)
                hidden_states = outputs.hidden_states
                idx = self.layer_idx if self.layer_idx < 0 else min(self.layer_idx, len(hidden_states) - 1)
                wavlm_features = hidden_states[idx] # B,L,C
                if idx < self.num_hidden_layers and idx !=-1:
                    wavlm_features = self.post_wavlm_layer_norm(wavlm_features)
                hidden_states = wavlm_features.detach().clone()
        else:
            with torch.no_grad():
                teacher_outputs = self.model(input_values=inputs.input_values.squeeze(0).to(device), output_hidden_states=True)
                teacher_hidden_states = teacher_outputs.hidden_states
                idx = self.layer_idx if self.layer_idx < 0 else min(self.layer_idx, len(teacher_hidden_states) - 1)
                teacher_wavlm_features = teacher_hidden_states[idx]
                if idx < self.num_hidden_layers and idx !=-1:
                    teacher_wavlm_features = self.post_wavlm_layer_norm(teacher_wavlm_features)
                teacher_wavlm_features = teacher_wavlm_features.detach() # B,L,C
                
            outputs = self.rep_model(input_values=inputs.input_values.squeeze(0).to(device), output_hidden_states=True)
            hidden_states = outputs.hidden_states
            idx = self.layer_idx if self.layer_idx < 0 else min(self.layer_idx, len(hidden_states) - 1)
            wavlm_features = hidden_states[idx] # B,L,C
            if idx < self.num_hidden_layers and idx!=-1:
                wavlm_features = self.post_wavlm_layer_norm(wavlm_features)
            hidden_states = wavlm_features.clone()
                
        position_bias = None
        attention_mask = None
        for i, layer in enumerate(self.enc_transformer):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                position_bias=position_bias,
            )
            hidden_states, position_bias = layer_outputs[:2]

        z = self.enc_projection(hidden_states)

        origin_len = z.size(1)
        
        if self.use_vae:
            mu = self.fc_mu(z)
            log_var = self.fc_var(z)
            log_var = torch.clamp(log_var, min=-12, max=12)
            
            z_hat = self.reparameterize(mu, log_var)
            kl_loss = self.compute_kl_loss(mu, log_var)
        else: # ae
            z_hat = z
            kl_loss = torch.tensor(0.0, device=device)

        hidden_states = self.dec_projection(z_hat)
        
        position_bias = None
        attention_mask = None
        for i, layer in enumerate(self.dec_transformer):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                position_bias=position_bias,
            )
            hidden_states, position_bias = layer_outputs[:2]
        
        if self.stage == 1:
            sr_loss = self.compute_semantic_reconstruction_loss(hidden_states, wavlm_features)
            sr_loss_recon = sr_loss_ssl = torch.tensor(0.0, device=hidden_states.device)
        else:
            sr_loss_recon = self.compute_semantic_reconstruction_loss(hidden_states, teacher_wavlm_features)
            sr_loss_ssl = self.compute_semantic_reconstruction_loss(wavlm_features, teacher_wavlm_features)
            sr_loss = sr_loss_recon + sr_loss_ssl
        return z_hat, kl_loss, sr_loss, sr_loss_recon, sr_loss_ssl
    
    def infer(self, audio, **kwargs):
        output_attentions=False
        device = audio.device
        inputs = self.processor(audio, sampling_rate=16000, padding=True, return_tensors="pt")

        if self.stage == 1:
            self.model.eval()
            outputs = self.model(input_values=inputs.input_values.squeeze(0).to(device), output_hidden_states=True)
        else:
            self.rep_model.eval()
            outputs = self.rep_model(input_values=inputs.input_values.squeeze(0).to(device), output_hidden_states=True)
        hidden_states = outputs.hidden_states
        idx = self.layer_idx if self.layer_idx < 0 else min(self.layer_idx, len(hidden_states) - 1)
        wavlm_features = hidden_states[idx] # B,L,C
        if idx < self.num_hidden_layers and idx !=-1:
            wavlm_features = self.post_wavlm_layer_norm(wavlm_features)            
        hidden_states = wavlm_features
        
        position_bias = None
        attention_mask = None
        for i, layer in enumerate(self.enc_transformer):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                position_bias=position_bias,
            )
            hidden_states, position_bias = layer_outputs[:2]
            
        z = self.enc_projection(hidden_states)
        if self.use_vae:
            mu = self.fc_mu(z)
            log_var = self.fc_var(z)
            log_var = torch.clamp(log_var, min=-12, max=12)
            z_hat = self.reparameterize(mu, log_var)
        else: #ae
            z_hat = z
            
        return z_hat
        
    def compute_semantic_reconstruction_loss(self, recon_features, target_features):
        l2_loss = F.mse_loss(recon_features, target_features)

        cosine_sim = F.cosine_similarity(recon_features, target_features, dim=-1)
        cosine_loss = 1.0 - cosine_sim.mean()
        
        semantic_reconstruction_loss = l2_loss + cosine_loss
        return semantic_reconstruction_loss

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def compute_kl_loss(self, mu, log_var):
        kl_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - (log_var.exp() + 1e-6), dim=-1
        )
        return kl_loss.mean()