# Copyright 2025 Xiaomi Corporation.
""" Example Usage: see README.md
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import mimo_audio_tokenizer
import torch
import torch.distributed as dist
import torchaudio
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from mimo_audio_tokenizer.config import MiMoAudioTokenizerConfig
from mimo_audio_tokenizer.model import MiMoAudioTokenizer


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_file_async(reconstructed_wav, quantized_tokens, info, timing_stats, sample_rate=24000):
    """Save file asynchronously."""
    try:
        if reconstructed_wav is not None:
            os.makedirs(os.path.dirname(info['reconstructed_wav']), exist_ok=True)
            torchaudio.save(info['reconstructed_wav'],
                            reconstructed_wav.float().cpu().detach(),
                            sample_rate,
                            format='wav',
                            encoding='PCM_S')
            duration = reconstructed_wav.shape[-1] * 1.0 / sample_rate
        else:
            audio_info = torchaudio.info(info['wav'])
            duration = audio_info.num_frames / audio_info.sample_rate
        rtf = ((timing_stats['dataloader_time'] + timing_stats['model_inference_time']) /
               timing_stats['batch_size']) / duration
        timing_stats['rtf'] = rtf
        timing_stats['duration'] = duration
        info['timing_stats'] = timing_stats
        if quantized_tokens is not None:
            os.makedirs(os.path.dirname(info['quantized_tokens']), exist_ok=True)
            for_save = {"quantized_tokens": quantized_tokens.cpu().detach().numpy().tolist(), "info": info}
            with open(info['quantized_tokens'], "w") as f:
                json.dump(for_save, f, ensure_ascii=False, indent=4)
        return duration
    except Exception as e:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        tqdm.write(f"[{timestamp}] - [ERROR] - Error saving audio {info.get('key', 'unknown')}: {e}")
        return 0.0


class MiMoAudioTokenizerDataset(Dataset):

    def __init__(self, data_list, task, model_config: MiMoAudioTokenizerConfig):
        self.datas = []
        self.task = task
        self.model_config = model_config
        """Example data_list:
        ```
        {"key": "uttid_1", "wav": "/mnt/data/audio/uttid_1.wav", "quantized_tokens": "/mnt/data/audio_reconstructed/uttid_1.json", "reconstructed_wav": "/mnt/data/audio_reconstructed/uttid_1.wav"}  # noqa
        ...
        {"key": "uttid_2", "wav": "/mnt/data/audio/uttid_2.wav", "quantized_tokens": "/mnt/data/audio_reconstructed/uttid_2.json", "reconstructed_wav": "/mnt/data/audio_reconstructed/uttid_2.wav"}  # noqa
        ...
        ```
        Note:
            - `key` is the key of this sample.
            - `wav` is the original audio.
            - `quantized_tokens` is the json path to save quantized tokens (we highly recommend to pre-define the save path before running the script).  # noqa
            - `reconstructed_wav` is the wav path to save reconstructed result (we highly recommend to pre-define the save path before running the script).  # noqa
        """
        missing = 0
        with open(data_list, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
            if torch.distributed.get_node_local_rank() == 0:
                iterator = tqdm(lines, desc='Loading data')
            else:
                iterator = lines
            for line in iterator:
                data = json.loads(line.strip())
                valid = True
                if task == "wav2token":
                    required_keys = ['key', 'wav', 'quantized_tokens']
                elif task == "token2wav":
                    required_keys = ['key', 'quantized_tokens', 'reconstructed_wav']
                elif task == "wav2token2wav":
                    required_keys = ['key', 'wav', 'quantized_tokens', 'reconstructed_wav']
                else:
                    raise ValueError(f"Invalid task: {task}")
                for k in required_keys:
                    if k not in data:
                        valid = False
                        break
                    if data[k] is None:
                        valid = False
                        break
                    if k == 'wav' and not os.path.exists(data['wav']):
                        valid = False
                        break
                if valid:
                    self.datas.append(data)
                else:
                    missing += 1
        if torch.distributed.get_node_local_rank() == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(
                f'[{timestamp}] - [INFO] - Loaded {total_lines} lines, found {missing} missing lines, total valid lines == {len(self.datas)}.'  # noqa
            )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        try:
            if self.task == "wav2token" or self.task == "wav2token2wav":
                wav = mimo_audio_tokenizer.load_audio(data['wav'], sr=self.model_config.sampling_rate)  # [T]
                mel = mimo_audio_tokenizer.mel_spectrogram(wav, self.model_config)  # [num_mels, T]
                return {"mel": mel, "tokens": None, "info": data}
            elif self.task == "token2wav":
                tokens = json.load(open(data['quantized_tokens'], 'r'))['quantized_tokens']
                return {"mel": None, "tokens": tokens, "info": data}
            else:
                raise ValueError(f"Invalid task: {self.task}")
        except Exception as e:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [WARNING] - Error processing data item {data.get('key', idx)}: {e}")
            return None


def collate_fn(batch):
    mels = [item["mel"] for item in batch if item is not None]
    mels_lens = [len(mel) for mel in mels if mel is not None]
    if len(mels) > 0 and mels[0] is not None:
        mels, mels_lens = mimo_audio_tokenizer.padding(mels)  # [B, num_mels, T]
    tokens = [item["tokens"] for item in batch if item is not None]
    tokens_lens = [len(token) for token in tokens if token is not None]
    if len(tokens) > 0 and tokens[0] is not None:
        tokens = [torch.tensor(token, dtype=torch.int32) for token in tokens]
        tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
        tokens_lens = torch.tensor(tokens_lens, dtype=torch.int32)
    infos = [item["info"] for item in batch if item is not None]
    return {
        "mels": mels,
        "mels_lens": mels_lens,
        "codes": tokens,
        "codes_lens": tokens_lens,
        "infos": infos,
    }


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    tqdm.write(
        f'[{timestamp}] - [INFO] - Inference on multiple gpus, this gpu {local_rank}, rank {rank}, world_size {world_size}'  # noqa
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def get_args():
    parser = argparse.ArgumentParser(description='MimoAudioTokenizer')
    parser.add_argument('--model_path', required=True, type=str, help='model path')
    parser.add_argument('--data_list', required=True, type=str, help='data list')
    parser.add_argument('--batch_size', required=True, type=int, help='batch size (per-device) for dataloading')
    parser.add_argument('--num_workers', type=int, default=4, help='workers for dataloader')
    parser.add_argument('--prefetch', type=int, default=5, help='prefetch for dataloader')
    parser.add_argument('--seed', type=int, default=1986, help='random seed for generation')
    parser.add_argument('--num_quantizers', type=int, default=20, help='number of quantizers')
    parser.add_argument('--task',
                        type=str,
                        choices=["wav2token", "token2wav", "wav2token2wav"],
                        default="wav2token",
                        help='task to perform')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    assert (torch.cuda.is_available())
    world_size, local_rank, rank = init_distributed()

    set_all_random_seed(args.seed)

    model = MiMoAudioTokenizer.from_pretrained(args.model_path, process=False)
    model.eval().bfloat16().cuda()
    dataset = MiMoAudioTokenizerDataset(args.data_list, args.task, model.config)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            sampler=sampler,
                            shuffle=False,
                            prefetch_factor=args.prefetch,
                            collate_fn=collate_fn)
    total_steps = len(dataset)

    if local_rank == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        tqdm.write(f"[{timestamp}] - [INFO] - {args}")
        progress_bar = tqdm(total=total_steps,
                            desc="Processing samples",
                            unit="wav",
                            position=0,
                            leave=True,
                            dynamic_ncols=True)

    cpu_counts = os.cpu_count()
    executor = ThreadPoolExecutor(max_workers=min(args.batch_size, cpu_counts // 8))
    pending_futures = []
    dataloader_iter = iter(dataloader)
    succeed_duration = 0.01  # avoid division by zero
    start_time = time.time()
    estimated_total_wavs = 0
    succeed_wavs = 0
    failed_wavs = 0
    last_print_time = start_time

    while True:
        try:
            dataloader_start = time.time()
            batch = next(dataloader_iter)
            dataloader_time = time.time() - dataloader_start

            if len(batch['infos']) == 0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                tqdm.write(
                    f"[{timestamp}] - [WARNING] - rank {rank} of {world_size}: No valid batch found, skipping this batch..."  # noqa
                )
                continue

            model_start = time.time()
            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].cuda()
            if args.task == "wav2token":
                tokens, tokens_lens, timing_stats = model.encode(**batch, n_q=args.num_quantizers)
                wavs, wavs_lens = [None] * timing_stats['batch_size'], [0] * timing_stats['batch_size']
            elif args.task == "token2wav":
                wavs, wavs_lens, timing_stats = model.decode(**batch)
                tokens, tokens_lens = [None] * timing_stats['batch_size'], [0] * timing_stats['batch_size']
            elif args.task == "wav2token2wav":
                tokens, tokens_lens, _timing_stats = model.encode(**batch, n_q=args.num_quantizers)
                wavs, wavs_lens, timing_stats = model.decode(tokens, tokens_lens)
                timing_stats.update(_timing_stats)
            else:
                raise ValueError(f"Invalid task: {args.task}")
            model_time = time.time() - model_start

            estimated_total_wavs += timing_stats['batch_size']

            timing_stats['dataloader_time'] = dataloader_time
            timing_stats['model_inference_time'] = model_time

            for i in range(timing_stats['batch_size']):
                # Handle different data types for wavs and tokens based on task
                if isinstance(wavs, list):
                    wav_data = wavs[i]  # wavs is a list of None or tensor
                else:
                    wav_data = wavs[i, :, :wavs_lens[i]]  # wavs is a tensor
                if isinstance(tokens, list):
                    token_data = tokens[i]  # tokens is a list of None or tensor
                else:
                    token_data = tokens[i, :tokens_lens[i], :]  # tokens is a tensor
                future = executor.submit(save_file_async, wav_data, token_data, batch['infos'][i].copy(),
                                         timing_stats.copy())
                pending_futures.append(future)

            completed_futures = []
            for future in pending_futures:
                if future.done():
                    try:
                        duration = future.result()
                        succeed_duration += duration
                        succeed_wavs += 1
                    except Exception as e:
                        failed_wavs += 1
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                        tqdm.write(
                            f"[{timestamp}] - [ERROR] - rank {rank} of {world_size}: Error in async save task: {e}")
                    completed_futures.append(future)

            for future in completed_futures:
                pending_futures.remove(future)

            if local_rank == 0:
                update_n = world_size * timing_stats['batch_size']
                if progress_bar.n + update_n > progress_bar.total:
                    progress_bar.update(progress_bar.total - progress_bar.n)
                else:
                    progress_bar.update(update_n)

                current_time = time.time()
                if current_time - last_print_time >= 120:
                    elapsed_time = current_time - start_time
                    avg_duration = succeed_duration / succeed_wavs if succeed_wavs > 0 else 0
                    estimated_total_duration = avg_duration * estimated_total_wavs
                    current_rtf = elapsed_time / estimated_total_duration if estimated_total_duration > 0.01 else 0
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                    tqdm.write(
                        f"[{timestamp}] - [INFO] - rank {rank} of {world_size}: Estimated total wavs: {estimated_total_wavs} ({estimated_total_wavs - succeed_wavs} pending to save), Succeed wavs: {succeed_wavs}, Failed wavs: {failed_wavs}, Estimated total duration: {estimated_total_duration:.2f}s ({estimated_total_duration / 3600:.2f} h), Estimated RTF: {current_rtf:.5f}, Elapsed time: {elapsed_time:.2f}s"  # noqa
                    )
                    last_print_time = current_time
        except StopIteration:
            break
        except Exception as e:
            failed_wavs += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [ERROR] - rank {rank} of {world_size}: Error in main loop: {e}")
            continue

    total_time = time.time() - start_time

    if local_rank == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        tqdm.write(f"[{timestamp}] - [INFO] - Waiting for {len(pending_futures)} pending save tasks to complete...")

    for future in pending_futures:
        try:
            duration = future.result(timeout=60)
            succeed_duration += duration
            succeed_wavs += 1
        except Exception as e:
            failed_wavs += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [ERROR] - rank {rank} of {world_size}: Error in final async save task: {e}")
    executor.shutdown(wait=True)

    if local_rank == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        tqdm.write(f"[{timestamp}] - [INFO] - All async save tasks completed.")
        progress_bar.close()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    tqdm.write(
        f"[{timestamp}] - [INFO] - rank {rank} of {world_size}: Final Report - Succeed wavs: {succeed_wavs}, Failed wavs: {failed_wavs}, Total duration: {succeed_duration:.2f}s ({succeed_duration / 3600:.2f} h), RTF: {total_time / succeed_duration:.5f}"  # noqa
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
