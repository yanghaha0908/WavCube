# WavCube: Unifying Speech Representation for Understanding and Generation via Semantic-Acoustic Joint Modeling

[![github](https://img.shields.io/badge/Code-Repo-black?logo=github)](https://github.com/yanghaha0908/WavCube)
[![arXiv](https://img.shields.io/badge/%F0%9F%93%84%20ArXiv-Paper-red.svg)](https://arxiv.org/abs/2605.06407)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20WavCube-Models-blueviolet)](https://huggingface.co/yhaha/WavCube)


WavCube is a 128-dim, 50Hz continuous representation that unifies speech understanding,
reconstruction, and generation within a single space.
This is the official code for the paper [WavCube: Unifying Speech Representation for Understanding and Generation via Semantic-Acoustic Joint Modeling](https://arxiv.org/pdf/2605.06407) [[abs](https://arxiv.org/abs/2605.06407)].

<!-- ## ✨ Key Features
- **[Feature 1]** – description.
- **[Feature 2]** – description.
- **[Feature 3]** – description.

## 🎧 Samples
To get a quick sense of our model's performance, please check out the [Demo Page](https://[YOUR_DEMO_PAGE]). **Comprehensive benchmarks covering a variety of baselines are available on this page.** -->

<!-- ## 🛠️ Installation

We recommend creating a fresh conda environment for installation.

### Env Setup
```bash
conda create -n WavCube python=3.9 -y
conda activate WavCube
```

### Basic Requirements
Please change `cu121` in the command to match your local CUDA version.
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja cmake pybind11 pytorch_lightning transformers
conda install -c conda-forge 'ffmpeg<7'
```

Afterwards, you can clone this repository and install the remaining dependencies:
```bash
git clone https://github.com/[YOUR_GITHUB_REPO].git
cd WavCube
pip install -r requirements.txt
``` -->

## 🚀 Quick Start

### Checkpoint Download
Pre-trained model checkpoints are available. Please use the following links to download the checkpoints:

| Representation | Dimension | Sample Rate | Frame Rate |
|----------------|-----------|-------------|------------|
| 🤗 [WavCube](https://huggingface.co/yhaha/WavCube/tree/main/WavCube) | 128 | 16k Hz | 50 Hz |
| 🤗 [WavCube-pro](https://huggingface.co/yhaha/WavCube/tree/main/WavCube-Pro) | 128 | 16k Hz | 50 Hz |


### Extract Representation from Speech
You can get continuous representations from raw wav using the following code:

```bash
python wav_to_feature.py \
    --audio 19_198_000000_000002.wav \
    --config configs/WavCube-stage2.yaml \
    --ckpt WavCube/checkpoints/vocos_checkpoint_epoch=177_step=195000_val_loss=3.3080.ckpt \
    --output 19_198_000000_000002.pt
```

### Reconstruct Speech from Representation

You can reconstruct waveform from representations using the following code:

```bash
python feature_to_wav.py \
    --feature 19_198_000000_000002.pt \
    --config configs/WavCube-stage2.yaml \
    --ckpt WavCube/checkpoints/vocos_checkpoint_epoch=177_step=195000_val_loss=3.3080.ckpt
```

<!-- ## 💡 Tips
- For devices that do not support BF16, you can manually disable PyTorch's mixed precision manager.
- If you encounter any issues or have questions, please feel free to open an issue. -->

## ❤️ Acknowledgements

We sincerely thank the authors of the following open-source projects, whose excellent work laid the foundation for WavCube: [Vocos](https://github.com/gemelo-ai/vocos), [Semantic-VAE](https://github.com/ZhikangNiu/Semantic-VAE), [MiMo-Audio-Tokenizer](https://github.com/XiaomiMiMo/MiMo-Audio-Tokenizer).



## 📝 Citation

If you find this repo helpful, please cite our work:

```bibtex
@misc{[CITATION_KEY],
      title={[Paper Title Placeholder]},
      author={[Author List]},
      year={2025},
      eprint={[ARXIV_ID]},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/[ARXIV_ID]},
}
```

## 📄 License

The code in this repository is released under the MIT license, see [LICENSE](LICENSE) for details.
