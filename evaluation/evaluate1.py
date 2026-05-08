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
from tqdm import tqdm
import soundfile as sf
import numpy as np
import torchaudio
from pathlib import Path
import soxr

# 假设该模块在同级目录下，如果没有改动该文件，确保 list_infer 支持 ref_texts=None
from speech_evaluator import SpeechQualityEvaluator

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

def parse_args():
    parser = argparse.ArgumentParser(description="Wav reconstruct quality evaluation.")

    # 设置默认路径为你提供的路径
    default_rec_path = "/inspire/hdd/global_user/yangguanrou-253108120172/codes/vocos/librispeech_test_clean/syn"
    default_ref_path = "/inspire/hdd/global_user/yangguanrou-253108120172/codes/vocos/librispeech_test_clean/gt"

    parser.add_argument("--rec_path", type=str, default=default_rec_path, help="Path to reconstructed wavs (Syn).")
    parser.add_argument("--ref_path", type=str, default=default_ref_path, help="Path to original wavs (GT).")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    rec_root = Path(args.rec_path)
    ref_root = Path(args.ref_path)
    
    # 检查路径是否存在
    if not rec_root.exists():
        raise FileNotFoundError(f"Reconstructed path not found: {rec_root}")
    if not ref_root.exists():
        raise FileNotFoundError(f"Reference path not found: {ref_root}")

    inputs, results = [], []
    
    # 获取 GT 目录下所有的 flac 文件 (支持子目录递归，或者扁平结构)
    # 如果你的文件是扁平的（没有子文件夹），rglob 也能工作
    ref_files = sorted(list(ref_root.rglob("*.flac")))
    
    print(f"Found {len(ref_files)} reference files in {ref_root}")

    Evaluator = SpeechQualityEvaluator()
    
    # 目标采样率，通常评估指标(PESQ等)需要16k
    TARGET_SR = 16000 

    for ref_file_path in tqdm(ref_files, desc='Processing audios'):
        # 获取相对路径，以便在 rec_root 中找到对应的文件
        # 例如：如果 ref 是 .../gt/1089/134/1089-134-0001.flac
        # rel_path 就是 1089/134/1089-134-0001.flac
        rel_path = ref_file_path.relative_to(ref_root)
        
        # 构造 rec 文件路径
        rec_file_path = rec_root / rel_path
        
        # 如果 rec 那边文件名可能是 .wav 而不是 .flac，可以取消下面这行的注释进行尝试
        # if not rec_file_path.exists(): rec_file_path = rec_file_path.with_suffix('.wav')

        if not rec_file_path.exists():
            print(f"Warning: Missing reconstructed file for {rel_path}, skipping.")
            continue
        
        # 读取音频
        wav_raw, raw_sr = sf.read(ref_file_path)
        wav_rec, rec_sr = sf.read(rec_file_path)

        # 转单声道 & float32
        wav_raw = _to_mono_float32(wav_raw)
        wav_rec = _to_mono_float32(wav_rec)

        # 统一重采样到 16k 用于评估
        if raw_sr != TARGET_SR:
            wav_raw = _resample_np(wav_raw, raw_sr, TARGET_SR)
        
        if rec_sr != TARGET_SR:
            wav_rec = _resample_np(wav_rec, rec_sr, TARGET_SR)

        # 长度对齐 (以最短的为准，或者根据评估器要求)
        min_len = min(len(wav_raw), len(wav_rec))
        wav_raw = wav_raw[:min_len]
        wav_rec = wav_rec[:min_len]

        inputs.append(wav_raw)
        results.append(wav_rec)

    if len(inputs) == 0:
        print("No valid audio pairs found.")
        return

    print("Calculating metrics...")
    
    # 计算指标，传入 ref_texts=None 跳过 WER 计算
    # 注意：这里 sample_rate=16000 告诉 evaluator 输入数据的采样率
    metrics = Evaluator.list_infer(inputs, results, sample_rate=TARGET_SR, ref_texts=None)

    print(metrics)
    
    # 结果保存路径
    save_path = rec_root / "_metric_no_wer.json"
    with open(save_path, "w") as f:
        f.write(json.dumps(metrics, indent=4, ensure_ascii=False))
        print(f"Metrics saved to {save_path}")

if __name__ == "__main__":
    main()