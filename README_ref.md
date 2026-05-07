# MagiCodec: Simple Masked Gaussian-Injected Audio Codec for High-Fidelity Reconstruction & Generation

[![github](https://img.shields.io/badge/Code-Repo-black?logo=github)](https://github.com/Ereboas/MagiCodec)
[![arXiv](https://img.shields.io/badge/%F0%9F%93%84%20ArXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2506.00385)
[![demo](https://img.shields.io/badge/%F0%9F%94%97%20MagiCodec-Demo-blue)](https://ereboas.github.io/MagiCodec/) 
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20MagiCodec-Models-blueviolet)](https://huggingface.co/Ereboas/MagiCodec_16k_50hz)

MagiCodec is a **single-layer**, **streaming** codec model that delivers state-of-the-art audio quality *and* highly model-able discrete tokens. 

This is the code for the MagiCodec neural audio codec presented in the paper [MagiCodec: Simple **Ma**sked **G**aussian-**I**njected Audio **Codec** for High-Fidelity Reconstruction and Generation](https://arxiv.org/pdf/2506.00385) [[abs](https://arxiv.org/abs/2506.00385)].

## ✨ Key Features
- **Single‑layer streaming design** – lightweight causal Transformer that drops straight into audio‑native LLMs with minimal latency.
- **Low bit‑rate & compute‑efficient** – <= 850 bps and <= 40 ms look‑ahead keep bandwidth and FLOPs tiny for on‑device use.
- **Rich acoustic *and* semantic information** – captures fine‑grained detail plus high‑level semantics, achieving top scores in both waveform reconstruction and downstream tasks.

## 🎧 Samples
To get a quick sense of our codec’s performance, please check out the [Sample Page](). **Comprehensive benchmarks covering a variety of baselines are available on this page.**

## 🛠️ Installation

Our released code is based on a specific version ([2.3.6](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.3.6)) of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/92dd5703ecdb99aa4a4aee9817f28557907403a2). 

Please note that different (including newer or older) versions of flash-attention may introduce changes to the C/Python interfaces called by our custom ops.
Compatibility issues may arise if you use a different version of flash-attention.

We recommend creating a fresh conda environment for installation.

### Env Setup
```bash
conda create -n MagiCodec python=3.9 -y
conda activate MagiCodec
```
### Basic Requirements
Please change `cu121` in the command to match your local CUDA version.
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja cmake pybind11 pytorch_lightning transformers
conda install -c conda-forge 'ffmpeg<7'  
````

### Flash-Attention Build
If your environment (including PyTorch, Python, and CUDA versions) matches one of the prebuilt wheels provided in the [flash-attention 2.3.6 release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.3.6), you can directly install flash-attention from the corresponding wheel. Otherwise, we recommend building flash-attention from source manually for best compatibility. The following script can be used to build flash-attention manually:

```bash
# flash-attn build
git clone https://github.com/Dao-AILab/flash-attention.git
git checkout 92dd570
git submodule sync --recursive
git submodule update --init --recursive
cd flash-attention
python setup.py install

# ops build
cd csrc
cd rotary && python setup.py install && cd ..
cd layer_norm && python setup.py install && cd ..
cd fused_dense_lib && python setup.py install && cd ..
```

Afterwards, you can clone this repository, and enjoy using MagiCodec!


## 🚀 Quick Start

### Checkpoint Download
Pre-trained model checkpoints are available. Please use the following links to download the checkpoints:

| Model Name               |    Dataset    |  Sample Rate  | Token Rate 
|--------------------------|---------------|---------------|------------|
| 🤗 [MagiCodec-50Hz-Base](https://huggingface.co/Ereboas/MagiCodec_16k_50hz)  |   Librilight  |    16k Hz     |   50 Hz


### Inference from raw wav
You can get discrete tokens and reconstructed waveform from raw wav using the following code:

```python
from codec.generator import Generator
import torch
import torchaudio
torch.set_grad_enabled(False)

target_sr = 16000
token_hz = 50
model_path = "MagiCodec-50Hz-Base.ckpt"

model = Generator(
    sample_rate = target_sr,
    token_hz = token_hz,
)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

device = f"cuda:0"
model = model.to(device)
model.eval()

def preprocess(path):
    x, sr = torchaudio.load(path)
    x = x.to(device)
    x = x.mean(dim=0, keepdim=True)
    x = torchaudio.functional.resample(x, sr, target_sr)
    return x[None, ...]

def infer(path):
    x = preprocess(path)
    orig_length = x.shape[-1]

    recon, codes, zq = model.infer(x)
    recon = recon[..., :orig_length]
    return recon, codes

recon, codes = infer("audio/1580-141083-0000.flac")
torchaudio.save("recon.wav", recon[0].cpu(), target_sr)
```


### Inference from codes

You can also get reconstructed waveform from discrete tokens using the following code:

```python
def infer_from_code(codes, codebook):
    assert codes.ndim == 2   # (B, T)
    z_q = torch.nn.functional.embedding(codes, codebook)
    recon = model.decoder(z_q).float()
    return recon

with torch.autocast(
    device_type = "cuda",
    dtype = torch.bfloat16,
    enabled = True,
):
    codebook = model.quantizer.codebook_proj(model.quantizer.codebook.weight) 
    recon_from_code = infer_from_code(codes, codebook)

torchaudio.save("recon_from_code.wav", recon_from_code[0].cpu(), target_sr)

print((recon==recon_from_code).all().item())
```

## 💡 Tips
- For devices that do not support BF16, you can manually disable PyTorch’s mixed precision manager.
- If you encounter any issues or have questions, please feel free to open an issue.

## 📝 Citation

If you find this repo helpful, please cite our work:

```bibtex
@misc{song2025magicodec,
      title={MagiCodec: Simple Masked Gaussian-Injected Codec for High-Fidelity Reconstruction and Generation}, 
      author={Yakun Song and Jiawei Chen and Xiaobin Zhuang and Chenpeng Du and Ziyang Ma and Jian Wu and Jian Cong and Dongya Jia and Zhuo Chen and Yuping Wang and Yuxuan Wang and Xie Chen},
      year={2025},
      eprint={2506.00385},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2506.00385}, 
}
```


## 📄 License

The code in this repository is released under the MIT license, see [LICENSE](LICENSE) for details.