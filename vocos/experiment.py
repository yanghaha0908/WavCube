import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
import transformers

from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.feature_extractors import FeatureExtractor
from vocos.heads import FourierHead
from vocos.helpers import plot_spectrogram_to_numpy
from vocos.loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss, MelSpecReconstructionLoss
from vocos.models import Backbone
from vocos.modules import safe_log
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)
from evaluation.ecapa_tdnn import ECAPA_TDNN_SMALL

class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        sample_rate: int,
        initial_learning_rate: float,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
        evaluate_stoi: bool = False,
        evaluate_pesq_wb: bool = False,
        evaluate_sim: bool = False,
    ):
        """
        Args:
            feature_extractor (FeatureExtractor): An instance of FeatureExtractor to extract features from audio signals.
            backbone (Backbone): An instance of Backbone model.
            head (FourierHead):  An instance of Fourier head to generate spectral coefficients and reconstruct a waveform.
            sample_rate (int): Sampling rate of the audio signals.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            num_warmup_steps (int): Number of steps for the warmup phase of learning rate scheduler. Default is 0.
            mel_loss_coeff (float, optional): Coefficient for Mel-spectrogram loss in the loss function. Default is 45.
            mrd_loss_coeff (float, optional): Coefficient for Multi Resolution Discriminator loss. Default is 1.0.
            pretrain_mel_steps (int, optional): Number of steps to pre-train the model without the GAN objective. Default is 0.
            decay_mel_coeff (bool, optional): If True, the Mel-spectrogram loss coefficient is decayed during training. Default is False.
            evaluate_utmos (bool, optional): If True, UTMOS scores are computed for each validation run.
            evaluate_pesq (bool, optional): If True, PESQ scores are computed for each validation run.
            evaluate_periodicty (bool, optional): If True, periodicity scores are computed for each validation run.
        """
        super().__init__()
        #self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head"])
        self.save_hyperparameters(
            "sample_rate",
            "initial_learning_rate",
            "num_warmup_steps",
            "mel_loss_coeff",
            "mrd_loss_coeff",
            "pretrain_mel_steps",
            "decay_mel_coeff",
            "evaluate_utmos",
            "evaluate_pesq",
            "evaluate_periodicty",
            "evaluate_stoi",
            "evaluate_pesq_wb",
            "evaluate_sim",
        )
        
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresddisc = MultiResolutionDiscriminator()

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

        self.train_discriminator = False
        self.base_mel_coeff = self.mel_loss_coeff = mel_loss_coeff

    def configure_optimizers(self):
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()},
        ]
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.backbone.parameters()},
            {"params": self.head.parameters()},
        ]

        opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )

        return (
            [opt_disc, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )

    def forward(self, audio_input, **kwargs):
        features = self.feature_extractor(audio_input, **kwargs)
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)
        return audio_output

    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):
        audio_input = batch

        # train discriminator
        if optimizer_idx == 0 and self.train_discriminator:
            with torch.no_grad():
                audio_hat = self(audio_input, **kwargs)

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)
            return loss

        # train generator
        if optimizer_idx == 1:
            audio_hat = self(audio_input, **kwargs)
            if self.train_discriminator:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

                self.log("generator/multi_period_loss", loss_gen_mp)
                self.log("generator/multi_res_loss", loss_gen_mrd)
                self.log("generator/feature_matching_mp", loss_fm_mp)
                self.log("generator/feature_matching_mrd", loss_fm_mrd)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

            mel_loss = self.melspec_loss(audio_hat, audio_input)
            loss = (
                loss_gen_mp
                + self.hparams.mrd_loss_coeff * loss_gen_mrd
                + loss_fm_mp
                + self.hparams.mrd_loss_coeff * loss_fm_mrd
                + self.mel_loss_coeff * mel_loss
            )

            self.log("generator/total_loss", loss, prog_bar=True)
            self.log("mel_loss_coeff", self.mel_loss_coeff)
            self.log("generator/mel_loss", mel_loss)

            if self.global_step % 1000 == 0 and self.global_rank == 0:
                self.logger.experiment.add_audio(
                    "train/audio_in", audio_input[0].data.cpu(), self.global_step, self.hparams.sample_rate
                )
                self.logger.experiment.add_audio(
                    "train/audio_pred", audio_hat[0].data.cpu(), self.global_step, self.hparams.sample_rate
                )
                with torch.no_grad():
                    mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                    mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
                self.logger.experiment.add_image(
                    "train/mel_target",
                    plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    "train/mel_pred",
                    plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )

            return loss

    def on_validation_epoch_start(self):
        # if self.hparams.evaluate_utmos:
        #     from metrics.UTMOS import UTMOSScore

        #     if not hasattr(self, "utmos_model"):
        #         self.utmos_model = UTMOSScore(device=self.device)
        
        if self.hparams.evaluate_utmos:
            self.sr_utmos = 16000
            self.utmos = torch.hub.load("ckpts/hub/tarepan_SpeechMOS_v1.2.0", "utmos22_strong", trust_repo=True, source="local").to(self.device)
        if self.hparams.evaluate_stoi:
            self.sr_stoi = 16000
            self.stoi = ShortTimeObjectiveIntelligibility(self.sr_stoi)
              
        if self.hparams.evaluate_pesq_wb: #wenxi
            self.sr_pesq = 16000
            self.pesq_wb = PerceptualEvaluationSpeechQuality(self.sr_pesq, mode="wb")
            self.pesq_nb = PerceptualEvaluationSpeechQuality(self.sr_pesq, mode="nb")
        
        if self.hparams.evaluate_sim:
            self.sim_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
            state_dict = torch.load("ckpts/wavlm_large_finetune.pth", weights_only=True, map_location=lambda storage, loc: storage)
            self.sim_model.load_state_dict(state_dict["model"], strict=False)
            self.sim_model.to(self.device).eval()
            
    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input = batch
        audio_hat = self(audio_input, **kwargs)

        audio_16_khz = torchaudio.functional.resample(audio_input, orig_freq=self.hparams.sample_rate, new_freq=16000)
        audio_hat_16khz = torchaudio.functional.resample(audio_hat, orig_freq=self.hparams.sample_rate, new_freq=16000)

        if self.hparams.evaluate_periodicty:
            from metrics.periodicity import calculate_periodicity_metrics

            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(audio_16_khz, audio_hat_16khz)
        else:
            periodicity_loss = pitch_loss = f1_score = 0

        if self.hparams.evaluate_utmos:
            # utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
            utmos_score = self.utmos(audio_hat_16khz, self.sr_utmos).mean()
        else:
            utmos_score = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_pesq:
            from pesq import pesq

            pesq_score = 0
            for ref, deg in zip(audio_16_khz.cpu().numpy(), audio_hat_16khz.cpu().numpy()):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_score = torch.tensor(pesq_score)
        else:
            pesq_score = torch.zeros(1, device=self.device)
            
        if self.hparams.evaluate_pesq_wb: #PESQ 是 CPU 密集型
            pesq_nb = self.pesq_nb(audio_hat_16khz, audio_16_khz)
            pesq_wb = self.pesq_wb(audio_hat_16khz, audio_16_khz)
        else:
            pesq_nb = torch.zeros(1, device=self.device)
            pesq_wb = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_stoi:
            stoi = self.stoi(audio_hat_16khz, audio_16_khz)
        else:
            stoi = torch.zeros(1, device=self.device)  

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score)

        return {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "utmos_score": utmos_score,
            "pesq_score": pesq_score,
            "periodicity_loss": periodicity_loss,
            "pitch_loss": pitch_loss,
            "f1_score": f1_score,
            "pesq_nb_score": pesq_nb,
            "pesq_wb_score": pesq_wb,
            "stoi_score": stoi,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        }

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            *_, audio_in, audio_pred = outputs[0].values()
            self.logger.experiment.add_audio(
                "val_in", audio_in.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
            ) #batch的第一条音频
            self.logger.experiment.add_audio(
                "val_pred", audio_pred.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
            )
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            self.logger.experiment.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val_mel_hat",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        utmos_score = torch.stack([x["utmos_score"] for x in outputs]).mean()
        pesq_score = torch.stack([x["pesq_score"] for x in outputs]).mean()
        periodicity_loss = np.array([x["periodicity_loss"] for x in outputs]).mean()
        pitch_loss = np.array([x["pitch_loss"] for x in outputs]).mean()
        f1_score = np.array([x["f1_score"] for x in outputs]).mean()

        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/utmos_score", utmos_score, sync_dist=True)
        self.log("val/pesq_score", pesq_score, sync_dist=True)
        self.log("val/periodicity_loss", periodicity_loss, sync_dist=True)
        self.log("val/pitch_loss", pitch_loss, sync_dist=True)
        self.log("val/f1_score", f1_score, sync_dist=True)

    @property
    def global_step(self):
        """
        Override global_step so that it returns the total number of batches processed
        """
        return self.trainer.fit_loop.epoch_loop.total_batch_idx

    def on_train_batch_start(self, *args):
        if self.global_step >= self.hparams.pretrain_mel_steps:
            self.train_discriminator = True
        else:
            self.train_discriminator = False

    def on_train_batch_end(self, *args):
        def mel_loss_coeff_decay(current_step, num_cycles=0.5):
            max_steps = self.trainer.max_steps // 2
            if current_step < self.hparams.num_warmup_steps:
                return 1.0
            progress = float(current_step - self.hparams.num_warmup_steps) / float(
                max(1, max_steps - self.hparams.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        if self.hparams.decay_mel_coeff:
            self.mel_loss_coeff = self.base_mel_coeff * mel_loss_coeff_decay(self.global_step + 1)


    # 1. 存的时候剔除（为了未来）
    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("utmos.")]
        for k in keys_to_remove:
            del state_dict[k]

    # 2. 读的时候忽略（为了兼容过去）
    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("utmos.")]
        if keys_to_remove:
            # 仅在主进程打印提示
            if self.global_rank == 0:
                print(f"Ignored {len(keys_to_remove)} 'utmos' keys from checkpoint load.")
            for k in keys_to_remove:
                del state_dict[k]

        keys_to_remove_sim = [k for k in state_dict.keys() if k.startswith("sim_model.")]
        if keys_to_remove_sim:
            if self.global_rank == 0:
                print(f"Ignored {len(keys_to_remove_sim)} 'sim_model' keys from checkpoint load.")
            for k in keys_to_remove_sim:
                del state_dict[k]

class VocosEncodecExp(VocosExp):
    """
    VocosEncodecExp is a subclass of VocosExp that overrides the parent experiment to function as a conditional GAN.
    It manages an additional `bandwidth_id` attribute, which denotes a learnable embedding corresponding to
    a specific bandwidth value of EnCodec. During training, a random bandwidth_id is generated for each step,
    while during validation, a fixed bandwidth_id is used.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        sample_rate: int,
        initial_learning_rate: float,
        num_warmup_steps: int,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
    ):
        super().__init__(
            feature_extractor,
            backbone,
            head,
            sample_rate,
            initial_learning_rate,
            num_warmup_steps,
            mel_loss_coeff,
            mrd_loss_coeff,
            pretrain_mel_steps,
            decay_mel_coeff,
            evaluate_utmos,
            evaluate_pesq,
            evaluate_periodicty,
        )
        # Override with conditional discriminators
        self.multiperioddisc = MultiPeriodDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))
        self.multiresddisc = MultiResolutionDiscriminator(num_embeddings=len(self.feature_extractor.bandwidths))

    def training_step(self, *args):
        bandwidth_id = torch.randint(low=0, high=len(self.feature_extractor.bandwidths), size=(1,), device=self.device,)
        output = super().training_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_step(self, *args):
        bandwidth_id = torch.tensor([0], device=self.device)
        output = super().validation_step(*args, bandwidth_id=bandwidth_id)
        return output

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            *_, audio_in, _ = outputs[0].values()
            # Resynthesis with encodec for reference
            self.feature_extractor.encodec.set_target_bandwidth(self.feature_extractor.bandwidths[0])
            encodec_audio = self.feature_extractor.encodec(audio_in[None, None, :])
            self.logger.experiment.add_audio(
                "encodec", encodec_audio[0, 0].data.cpu().numpy(), self.global_step, self.hparams.sample_rate,
            )

        super().validation_epoch_end(outputs)

class MiMoWavLMVAEExp(VocosExp):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead = None,
        sample_rate: int = 16000,
        initial_learning_rate: float = 1e-4,
        num_warmup_steps: int = 5000,
        mel_loss_coeff: float = 45,
        mrd_loss_coeff: float = 1.0,
        kl_loss_coeff: float = 0.01,
        sr_loss_coeff: float = 1.0,
        gan_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
        evaluate_stoi: bool = False,
        evaluate_pesq_wb: bool = False,
        evaluate_sim: bool = False,
    ):
        super().__init__(
            feature_extractor,
            backbone,
            head,
            sample_rate,
            initial_learning_rate,
            num_warmup_steps,
            mel_loss_coeff,
            mrd_loss_coeff,
            pretrain_mel_steps,
            decay_mel_coeff,
            evaluate_utmos,
            evaluate_pesq,
            evaluate_periodicty,
            evaluate_stoi,
            evaluate_pesq_wb,
            evaluate_sim,
        )
        
        self.save_hyperparameters("kl_loss_coeff","sr_loss_coeff", "gan_loss_coeff")
        self.kl_loss_coeff = kl_loss_coeff
        self.sr_loss_coeff = sr_loss_coeff
        self.gan_loss_coeff = gan_loss_coeff
        
        
    def configure_optimizers(self):
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresddisc.parameters()},
        ]
        
        gen_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.backbone.parameters()},
        ]            

        opt_disc = torch.optim.AdamW(disc_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))
        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))

        max_steps = self.trainer.max_steps // 2  # Max steps per optimizer
        scheduler_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        ) #线性预热，余弦衰减
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )

        return (
            [opt_disc, opt_gen],
            [{"scheduler": scheduler_disc, "interval": "step"}, {"scheduler": scheduler_gen, "interval": "step"}],
        )
        
    def forward(self, audio_input, **kwargs):
        z_hat, kl_loss, semantic_reconstruction_loss, sr_loss_recon, sr_loss_ssl = self.feature_extractor(audio_input, **kwargs) # B,T -> B,D(1024),T/320
        
        if self.feature_extractor.stage == 1:
            audio_output = self.backbone(z_hat.detach(), **kwargs)
        else:
            audio_output = self.backbone(z_hat, **kwargs)
        if self.head is not None:
            audio_output = self.head(audio_output)
        audio_output = self._align_length(audio_output, seconds=audio_input.shape[-1]/self.hparams.sample_rate) #10.0
        return audio_output, kl_loss, semantic_reconstruction_loss, sr_loss_recon, sr_loss_ssl
    
    def _align_length(self, audio_output: torch.Tensor, seconds: float = 3.0) -> torch.Tensor:
        target_len = int(self.hparams.sample_rate * seconds)
        cur_len = audio_output.size(1)
        
        if cur_len > target_len:
            audio_output = audio_output[:, :target_len]
        elif cur_len < target_len:
            pad_len = target_len - cur_len
            audio_output = F.pad(audio_output, (0, pad_len))

        return audio_output
    
    def training_step(self, batch, batch_idx, optimizer_idx, **kwargs):
        audio_input = batch

        # train discriminator
        if optimizer_idx == 0 and self.train_discriminator:
            with torch.no_grad():
                audio_hat, kl_loss, semantic_reconstruction_loss, sr_loss_recon, sr_loss_ssl = self(audio_input, **kwargs)

            real_score_mp, gen_score_mp, _, _ = self.multiperioddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            real_score_mrd, gen_score_mrd, _, _ = self.multiresddisc(y=audio_input, y_hat=audio_hat, **kwargs,)
            loss_mp, loss_mp_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mp, disc_generated_outputs=gen_score_mp
            )
            loss_mrd, loss_mrd_real, _ = self.disc_loss(
                disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = self.hparams.gan_loss_coeff * (loss_mp + self.hparams.mrd_loss_coeff * loss_mrd)

            self.log("discriminator/total", loss, prog_bar=True)
            self.log("discriminator/multi_period_loss", loss_mp)
            self.log("discriminator/multi_res_loss", loss_mrd)
            return loss

        # train generator
        if optimizer_idx == 1:
            audio_hat, kl_loss, semantic_reconstruction_loss, sr_loss_recon, sr_loss_ssl = self(audio_input, **kwargs)
            if self.train_discriminator:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = self.multiperioddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresddisc(
                    y=audio_input, y_hat=audio_hat, **kwargs,
                )
                loss_gen_mp, list_loss_gen_mp = self.gen_loss(disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = self.feat_matching_loss(fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

                self.log("generator/multi_period_loss", loss_gen_mp)
                self.log("generator/multi_res_loss", loss_gen_mrd)
                self.log("generator/feature_matching_mp", loss_fm_mp)
                self.log("generator/feature_matching_mrd", loss_fm_mrd)
            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

            mel_loss = self.melspec_loss(audio_hat, audio_input)
            loss = (
                self.hparams.gan_loss_coeff * (
                    loss_gen_mp
                    + self.hparams.mrd_loss_coeff * loss_gen_mrd
                    + loss_fm_mp
                    + self.hparams.mrd_loss_coeff * loss_fm_mrd
                )
                + self.mel_loss_coeff * mel_loss
                + self.kl_loss_coeff * kl_loss
                + self.sr_loss_coeff * semantic_reconstruction_loss
            )

            self.log("generator/total_loss", loss, prog_bar=True)
            self.log("mel_loss_coeff", self.mel_loss_coeff, prog_bar=True)
            self.log("generator/mel_loss", mel_loss)
            self.log("generator/kl_loss", kl_loss, prog_bar=True)
            self.log("generator/semantic_reconstruction_loss", semantic_reconstruction_loss, prog_bar=True)
            self.log("generator/sr_loss_recon", sr_loss_recon, prog_bar=True)
            self.log("generator/sr_loss_ssl", sr_loss_ssl, prog_bar=True)

            if self.global_step % 1000 == 0 and self.global_rank == 0:
                self.logger.experiment.add_audio(
                    "train/audio_in", audio_input[0].data.cpu(), self.global_step, self.hparams.sample_rate
                )
                self.logger.experiment.add_audio(
                    "train/audio_pred", audio_hat[0].data.cpu(), self.global_step, self.hparams.sample_rate
                )
                with torch.no_grad():
                    mel = safe_log(self.melspec_loss.mel_spec(audio_input[0]))
                    mel_hat = safe_log(self.melspec_loss.mel_spec(audio_hat[0]))
                self.logger.experiment.add_image(
                    "train/mel_target",
                    plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    "train/mel_pred",
                    plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )

            return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        audio_input = batch
        audio_hat, kl_loss, semantic_reconstruction_loss, sr_loss_recon, sr_loss_ssl = self(audio_input, **kwargs)

        audio_16_khz = torchaudio.functional.resample(audio_input, orig_freq=self.hparams.sample_rate, new_freq=16000)
        audio_hat_16khz = torchaudio.functional.resample(audio_hat, orig_freq=self.hparams.sample_rate, new_freq=16000)

        if self.hparams.evaluate_periodicty:
            from metrics.periodicity import calculate_periodicity_metrics

            periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(audio_16_khz, audio_hat_16khz)
        else:
            periodicity_loss = pitch_loss = f1_score = 0

        if self.hparams.evaluate_utmos:
            # utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()
            utmos_score = self.utmos(audio_hat_16khz, self.sr_utmos).mean()
        else:
            utmos_score = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_pesq:
            from pesq import pesq

            pesq_score = 0
            for ref, deg in zip(audio_16_khz.cpu().numpy(), audio_hat_16khz.cpu().numpy()):
                pesq_score += pesq(16000, ref, deg, "wb", on_error=1)
            pesq_score /= len(audio_16_khz)
            pesq_score = torch.tensor(pesq_score)
        else:
            pesq_score = torch.zeros(1, device=self.device)
            
        if self.hparams.evaluate_pesq_wb:
            pesq_nb = self.pesq_nb(audio_hat_16khz, audio_16_khz)
            pesq_wb = self.pesq_wb(audio_hat_16khz, audio_16_khz)
        else:
            pesq_nb = torch.zeros(1, device=self.device)
            pesq_wb = torch.zeros(1, device=self.device)

        if self.hparams.evaluate_stoi:
            stoi = self.stoi(audio_hat_16khz, audio_16_khz)
        else:
            stoi = torch.zeros(1, device=self.device)  
        
        if getattr(self.hparams, "evaluate_sim", False):
            with torch.no_grad():
                ref_emb = self.sim_model(audio_16_khz)
                hyp_emb = self.sim_model(audio_hat_16khz)
                sim_score = F.cosine_similarity(ref_emb, hyp_emb).mean()
        else:
            sim_score = torch.zeros(1, device=self.device)

        mel_loss = self.melspec_loss(audio_hat.unsqueeze(1), audio_input.unsqueeze(1))
        total_loss = mel_loss + (5 - utmos_score) + (5 - pesq_score) + (1 - sim_score) + kl_loss + semantic_reconstruction_loss

        return {
            "val_loss": total_loss,
            "mel_loss": mel_loss,
            "kl_loss": kl_loss,
            "semantic_reconstruction_loss": semantic_reconstruction_loss,
            "sr_loss_recon": sr_loss_recon,
            "sr_loss_ssl": sr_loss_ssl,
            "utmos_score": utmos_score,
            "pesq_score": pesq_score,
            "periodicity_loss": periodicity_loss,
            "pitch_loss": pitch_loss,
            "f1_score": f1_score,
            "pesq_nb_score": pesq_nb,
            "pesq_wb_score": pesq_wb,
            "stoi_score": stoi,
            "sim_score": sim_score,
            "audio_input": audio_input[0],
            "audio_pred": audio_hat[0],
        }

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            *_, audio_in, audio_pred = outputs[0].values()
            self.logger.experiment.add_audio(
                "val_in", audio_in.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
            ) #batch的第一条音频
            self.logger.experiment.add_audio(
                "val_pred", audio_pred.data.cpu().numpy(), self.global_step, self.hparams.sample_rate
            )
            mel_target = safe_log(self.melspec_loss.mel_spec(audio_in))
            mel_hat = safe_log(self.melspec_loss.mel_spec(audio_pred))
            self.logger.experiment.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "val_mel_hat",
                plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()
        kl_loss = torch.stack([x["kl_loss"] for x in outputs]).mean()
        semantic_reconstruction_loss = torch.stack([x["semantic_reconstruction_loss"] for x in outputs]).mean()
        sr_loss_recon = torch.stack([x["sr_loss_recon"] for x in outputs]).mean()
        sr_loss_ssl = torch.stack([x["sr_loss_ssl"] for x in outputs]).mean()
        utmos_score = torch.stack([x["utmos_score"] for x in outputs]).mean()
        pesq_score = torch.stack([x["pesq_score"] for x in outputs]).mean()
        sim_score = torch.stack([x["sim_score"] for x in outputs]).mean()
        periodicity_loss = np.array([x["periodicity_loss"] for x in outputs]).mean()
        pitch_loss = np.array([x["pitch_loss"] for x in outputs]).mean()
        f1_score = np.array([x["f1_score"] for x in outputs]).mean()

        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val/mel_loss", mel_loss, sync_dist=True)
        self.log("val/kl_loss", kl_loss, sync_dist=True)
        self.log("val/semantic_reconstruction_loss", semantic_reconstruction_loss, sync_dist=True)
        self.log("val/sr_loss_recon", sr_loss_recon, sync_dist=True)
        self.log("val/sr_loss_ssl", sr_loss_ssl, sync_dist=True)
        self.log("val/utmos_score", utmos_score, sync_dist=True)
        self.log("val/pesq_score", pesq_score, sync_dist=True)
        self.log("val/sim_score", sim_score, sync_dist=True)
        self.log("val/periodicity_loss", periodicity_loss, sync_dist=True)
        self.log("val/pitch_loss", pitch_loss, sync_dist=True)
        self.log("val/f1_score", f1_score, sync_dist=True)