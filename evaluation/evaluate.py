# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import soundfile as sf
import numpy as np
import torchaudio
from pathlib import Path
import soxr

from speech_evaluator import SpeechQualityEvaluator
from ecapa_tdnn import ECAPA_TDNN_SMALL

def load_audio(
    adfile: Path,
    sampling_rate: int = None,
    length: int = None,
    segment_duration: int = None,
):
    r"""Load audio file with target sampling rate and lsength

    Args:
        adfile (Path): path to audio file.
        sampling_rate (int, optional): target sampling rate. Defaults to None.
        length (int, optional): target audio length. Defaults to None.
        volume_normalize (bool, optional): whether perform volume normalization. Defaults to False.
        segment_duration (int): random select a segment with duration of {segment_duration}s.
                                Defualt to None which means the whole audio will be used.

    Returns:
        audio (np.ndarray): audio
    """
    audio, sr = sf.read(adfile)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # print('audio', adfile, audio)
    audio = np.array(audio.squeeze())

    if sampling_rate is not None and sr != sampling_rate:
        audio = soxr.resample(audio, sr, sampling_rate, quality="VHQ")
        sr = sampling_rate
        
    # check the audio length
    if length is not None:
        assert abs(audio.shape[0] - length) < 1000, (
            f"{adfile} Audio length is {audio.shape[0]}, but target length is {length}"
        )
        if audio.shape[0] > length:
            audio = audio[:length]
        else:
            audio = np.pad(audio, (0, int(length - audio.shape[0])))
    return audio


def _to_mono_float32(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:  # (T, C)
        x = x.mean(axis=1)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x


def _resample_np(wav_np: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    if orig_sr == new_sr:
        return wav_np
    wav_t = torch.from_numpy(wav_np).unsqueeze(0)  # (1, T)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)
    wav_t = resampler(wav_t)  # (1, T')
    return wav_t.squeeze(0).contiguous().numpy()


def process_audio(wav_path, args_dict):
    """Return wav_in (np.float32), already hp-filtered & padded to hop multiple."""
    wav_raw = load_audio(
        wav_path,
        sampling_rate=args_dict["sample_rate"],
        volume_normalize=args_dict["volume_normalize"],
    )
    if wav_raw is None:
        raise ValueError(f"Failed to load audio from {wav_path}")

    wav_in = wav_raw

    return wav_in


def parse_args():
    parser = argparse.ArgumentParser(description="Wav reconstruct quality evaluation.")

    parser.add_argument("--rec_path", type=str, required=True, help="Path to reconstructed wavs.")
    parser.add_argument("--ref_path", type=str, required=True, help="Path to original wavs.")
    parser.add_argument("--ref_texts_path", type=str, default=None, help="Path to the reference texts file (jsonl format).")
    parser.add_argument("--sim_ckpt", type=str, default="ckpts/wavlm_large_finetune.pth")

    return parser.parse_args()


def main():
    args = parse_args()
    rec_root = Path(args.rec_path)
    ref_root = Path(args.ref_path)
    ref_texts_path = Path(args.ref_texts_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id_text_map = {}
    with open(ref_texts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            obj = json.loads(line)
            id_text_map[obj["index"]] = obj["text"]
            
    inputs, results, ref_texts = [], [], []
    sim_scores = []
    Evaluator = SpeechQualityEvaluator()

    # 1. 加载 Speaker Similarity 模型
    print(f"Loading SIM model from {args.sim_ckpt}...")
    sim_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(args.sim_ckpt, weights_only=True, map_location=lambda storage, loc: storage)
    sim_model.load_state_dict(state_dict["model"], strict=False)
    sim_model.to(device).eval()

    for id,text in tqdm(id_text_map.items(), desc='Processing audios'):
        folder1,folder2 = id.split('-')[0],id.split('-')[1]
        rec_path = rec_root / folder1 / folder2 /(id + ".wav")
        # rec_path = rec_root / folder1 / folder2 /(id + ".flac")
        ref_path = ref_root / folder1 / folder2 /(id + ".flac")
        
        wav_raw, raw_sr = sf.read(ref_path)
        wav_rec, rec_sr = sf.read(rec_path)

        wav_raw = _to_mono_float32(wav_raw)
        wav_rec = _to_mono_float32(wav_rec)

        if rec_sr != raw_sr:
            wav_rec = _resample_np(wav_rec, rec_sr, raw_sr)
            rec_sr = raw_sr

        # --- 现场计算 Speaker Similarity ---
        with torch.no_grad():
            t_raw = torch.from_numpy(wav_raw).unsqueeze(0).to(device)
            t_rec = torch.from_numpy(wav_rec).unsqueeze(0).to(device)
            emb_ref = sim_model(t_raw)
            emb_rec = sim_model(t_rec)
            sim = F.cosine_similarity(emb_ref, emb_rec)[0].item()
            sim_scores.append(sim)

        inputs.append(wav_raw)
        results.append(wav_rec)

        ref_texts.append(text)

    # caculate metrics
    metrics = Evaluator.list_infer(inputs, results, sample_rate=16000, ref_texts=ref_texts)

    # 5. 整合 SIM 结果
    if sim_scores:
        avg_sim = np.mean(sim_scores)
        std_sim = np.std(sim_scores)
        metrics["SIM"] = [float(avg_sim), float(std_sim)]

    # caculate sim #要改 #TODO
    # if os.path.exists(os.path.join(args_dict['rec_path'], 'speaker_embedding')):
    #     featfiles = os.listdir(os.path.join(args_dict['rec_path'], 'speaker_embedding'))
    #     sims = []
    #     for featfile in tqdm(featfiles, desc='load speaker embeddings'):
    #         if os.path.splitext(featfile)[-1] != '.pt': continue
    #         if featfile.endswith('_rec.pt'):
    #             index = '_'.join(os.path.splitext(featfile)[0].split('_')[:-1])
    #         else:
    #             index = os.path.splitext(featfile)[0]
    #         rec_path = os.path.join(args_dict['rec_path'],'speaker_embedding', featfile)
    #         ref_path = os.path.join(args_dict['ref_path'], 'speaker_embedding', f'{index}.pt')

    #         feat_raw  = torch.load(ref_path)
    #         feat = torch.load(rec_path)
    #         sim = torch.cosine_similarity(feat_raw.unsqueeze(0), feat.unsqueeze(0))
    #         sims.append(sim.item())
    #     sim = sum(sims) / len(sims)
    #     metrics.update({'SIM': sim})
    
    print(metrics)
    with open(f"{args.rec_path}/_metric2.json", "w") as f:
        f.write(json.dumps(metrics, indent=-4, ensure_ascii=False))
        print(f"Metrics saved to {args.rec_path}/_metric2.json")

if __name__ == "__main__":
    main()
