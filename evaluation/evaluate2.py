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

from speech_evaluator import SpeechQualityEvaluator
from ecapa_tdnn import ECAPA_TDNN_SMALL


def _to_mono_float32(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2: x = x.mean(axis=1)
    if x.dtype != np.float32: x = x.astype(np.float32)
    return x

def _resample_np(wav_np: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    if orig_sr == new_sr: return wav_np
    wav_t = torch.from_numpy(wav_np).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)
    return resampler(wav_t).squeeze(0).numpy()

def parse_args():
    parser = argparse.ArgumentParser(description="Wav reconstruct quality evaluation.")
    
    # 默认路径
    default_rec = "/inspire/hdd/global_user/yangguanrou-253108120172/codes/vocos/librispeech_test_clean/syn"
    default_ref = "/inspire/hdd/global_user/yangguanrou-253108120172/codes/vocos/librispeech_test_clean/gt"
    default_ckpt = "ckpts/wavlm_large_finetune.pth"

    parser.add_argument("--rec_path", type=str, default=default_rec)
    parser.add_argument("--ref_path", type=str, default=default_ref)
    parser.add_argument("--ref_texts_path", type=str, required=True, help="Path to the JSONL metadata file.")
    parser.add_argument("--sim_ckpt", type=str, default=default_ckpt)
    
    return parser.parse_args()

def main():
    args = parse_args()
    rec_root = Path(args.rec_path)
    ref_root = Path(args.ref_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 Speaker Similarity 模型
    print(f"Loading SIM model from {args.sim_ckpt}...")
    sim_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(args.sim_ckpt, weights_only=True, map_location=lambda storage, loc: storage)
    sim_model.load_state_dict(state_dict["model"], strict=False)
    sim_model.to(device).eval()

    # 2. 初始化质量评估器 (PESQ, STOI, WER)
    evaluator = SpeechQualityEvaluator()
    
    inputs, results, ref_texts = [], [], []
    sim_scores = []
    TARGET_SR = 16000

    # 3. 读取 JSONL 并处理扁平化目录下的文件
    with open(args.ref_texts_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing audios"):
        data = json.loads(line.strip())
        idx = data["index"]
        text = data["text"]

        # 构造文件路径（扁平结构：直接在目录下寻找 index.flac 或 index.wav）
        # 先找 .flac，找不到再找 .wav
        ref_file = ref_root / f"{idx}.flac"
        if not ref_file.exists(): ref_file = ref_root / f"{idx}.wav"
        
        rec_file = rec_root / f"{idx}.flac"
        if not rec_file.exists(): rec_file = rec_root / f"{idx}.wav"

        if not ref_file.exists() or not rec_file.exists():
            print(f"Warning: File pair for {idx} not found. Skipping.")
            continue

        # 读取音频
        wav_raw, sr_raw = sf.read(ref_file)
        wav_rec, sr_rec = sf.read(rec_file)

        # 预处理：单声道 + float32
        wav_raw = _to_mono_float32(wav_raw)
        wav_rec = _to_mono_float32(wav_rec)

        # --- 现场计算 Speaker Similarity ---
        with torch.no_grad():
            t_raw = torch.from_numpy(wav_raw).unsqueeze(0).to(device)
            t_rec = torch.from_numpy(wav_rec).unsqueeze(0).to(device)
            emb_ref = sim_model(t_raw)
            emb_rec = sim_model(t_rec)
            sim = F.cosine_similarity(emb_ref, emb_rec)[0].item()
            sim_scores.append(sim)

        # --- 准备 PESQ / STOI / WER 数据 ---
        # 截断对齐（PESQ必须长度一致）
        # min_len = min(len(wav_raw_16k), len(wav_rec_16k))
        # inputs.append(wav_raw_16k[:min_len])
        # results.append(wav_rec_16k[:min_len])# 本来就是一样长的
        inputs.append(wav_raw)
        results.append(wav_rec)
    
        ref_texts.append(text)

    # 4. 运行质量评估器 (包括内部 WER 计算)
    print("\nCalculating PESQ, STOI, and WER...")
    metrics = evaluator.list_infer(inputs, results, sample_rate=TARGET_SR, ref_texts=ref_texts)

    # 5. 整合 SIM 结果
    if sim_scores:
        avg_sim = np.mean(sim_scores)
        std_sim = np.std(sim_scores)
        metrics["SIM"] = [float(avg_sim), float(std_sim)]

    # 6. 打印并保存结果
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    print(json.dumps(metrics, indent=4))

    save_path = rec_root / "_metric_summary.json"
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"\nResults saved to: {save_path}")

if __name__ == "__main__":
    main()