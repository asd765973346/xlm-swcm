# Multilingual Seq2Seq Model with Interleaved Transformer Decoder

XLM-SWCM (Cross-lingual Language Model with Shared Weights Cross-lingual Modeling) is an innovative sequence-to-sequence model specifically designed to address the challenges of extremely low-resource languages. Our framework introduces a novel weight-sharing mechanism between encoder and decoder components, enabling effective knowledge transfer from multilingual encoders to generation tasks.This repository contains the implementation of a multilingual sequence-to-sequence model that leverages shared weights pretraining for extremely low-resource languages. The model combines CINO-v2-base encoder with a custom interleaved transformer decoder architecture.

## Supported Languages

Primary focus on Chinese minority languages:

* Tibetan (bo)
* Uyghur (ug)
* Kazakh (kk)
* Mongolian (mn)
* Chinese (zh)

## 📋 Table of Contents

- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Model Download](#-model-download)
- [Usage](#-usage)
- [File Structure](#-file-structure)
- [Citation](#-citation)
- [License](#-license)

## 🏗️ Model Architecture

The model features:

- **Encoder**: CINO-v2-base for multilingual understanding
- **Decoder**: Custom interleaved transformer with dual FFN layers
- **Hybrid Design**: Combines normal and custom decoder layers
- **Initialization**: Leverages pre-trained encoder weights for decoder initialization

### Key Components:

- `NormalDecoderLayer`: Standard transformer decoder layer
- `CustomDecoderLayer`: Modified decoder with interleaved FFN architecture
- `InterleavedTransformerDecoder`: Hybrid decoder combining both layer types
- `Seq2SeqModel`: Complete encoder-decoder architecture

## 🚀 Installation

### 1. Environment Setup

Create a conda environment:

```bash
conda create -n seq2seq python=3.8
conda activate seq2seq
```

### 2. Install PyTorch

Install PyTorch compatible with your GPU. Visit [PyTorch Official Website](https://pytorch.org/get-started/locally/) to get the appropriate command for your system.

**For CUDA 11.8:**

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**For CUDA 12.1:**

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**For CPU only:**

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 3. Install Transformers

```bash
pip install transformers>=4.21.0
pip install tokenizers>=0.13.0
```

### 4. Additional Dependencies

```bash
pip install torch-audio
pip install sentencepiece
```

## 📥 Model Download

### 1. Download Base Model (CINO v2)

Download the CINO v2 base model from Hugging Face:

**Option 1: Using huggingface_hub (recommended)**

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download model files
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='hfl/cino-base-v2', local_dir='./base')
"
```

**Option 2: Manual download**

```bash
# Create base directory
mkdir -p base
```

1. Visit: https://huggingface.co/hfl/cino-base-v2
2. Download all model files (config.json, pytorch_model.bin, tokenizer files, etc.)
3. Place all downloaded files in the `./base/` directory

**Option 3: Direct loading in code**

```python
# The model can also be loaded directly without local download
from transformers import XLMRobertaModel, XLMRobertaConfig
model = XLMRobertaModel.from_pretrained('hfl/cino-base-v2')
```

**Required files in `base/` directory:**

- `config.json`
- `pytorch_model.bin`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.txt`

### 2. Download Pre-trained Weights

Download the XLM-SWCM model weights:

```bash
# Create pretrained_model directory
mkdir -p pretrained_model

Download XLM-SWCM weights from Hugging Face (coming soon)
URL: https://huggingface.co/KEVVVV/xlm-swcm
Place the downloaded xlm-swcm.bin file in ./pretrained_model/
```

**Note**: The XLM-SWCM weights will be available on Hugging Face soon. Check back for updates.

## 📖 Usage

### Basic Inference

```python
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaConfig
from model import Seq2SeqModel

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./pretrained_model/xlm-swcm.bin"
xlm_model_path = "./base"

# Load configuration and model
config = XLMRobertaConfig.from_pretrained(xlm_model_path)
model = Seq2SeqModel(
    model_name_or_path=xlm_model_path,
    decoder_config=config,
    device=device,
    tgtlen=256,
    batchsize=1,
    teacher_forcing=0.0
)

# Load pre-trained weights
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(xlm_model_path)

# Example inference
sample_text = "Your input text here"
inputs = tokenizer(sample_text, return_tensors='pt', max_length=256, truncation=True)
with torch.no_grad():
    outputs = model.greedy_decode(inputs['input_ids'], inputs['attention_mask'])
```

### Advanced Usage with Beam Search

```python
# Beam search decoding
beam_size = 5
n_best = 3

with torch.no_grad():
    batch_hyp, batch_scores = model.beam_decode(
        src_seq=inputs['input_ids'],
        src_mask=inputs['attention_mask'],
        beam_size=beam_size,
        n_best=n_best
    )

# Process results
for hyp, scores in zip(batch_hyp, batch_scores):
    for h, s in zip(hyp, scores):
        decoded = tokenizer.decode(h, skip_special_tokens=True)
        print(f"Score: {s:.4f} | Text: {decoded}")
```

### Running the Example Script

```bash
python inference_example.py
```

## 📁 File Structure

```
your-project/
├── model.py                 # Main model implementation
├── inference_example.py     # Example inference script
├── base/                    # CINO v2 base model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── pretrained_model/        # Pre-trained weights
│   └── xlm-swcm.bin
├── transformer/             # Additional transformer utilities
│   ├── Constants.py
│   └── Beam.py
└── README.md
```

## 🔧 Model Configuration

### Key Parameters:

- `tgtlen`: Maximum target sequence length (default: 256)
- `batchsize`: Batch size for inference (default: 1)
- `teacher_forcing`: Teacher forcing ratio during training (0.0 for inference)
- `beam_size`: Number of beams for beam search (default: 5)
- `n_best`: Number of best hypotheses to return (default: 3)

### Decoder Architecture:

- Custom decoder layers with dual FFN structure
- Regular insertion of normal decoder layers every 3 custom layers
- Encoder weight initialization for improved convergence

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@article{swcm,
  author       = {Zeli Su and Ziyin Zhang and Guixian Xu and Jianing Liu and XU Han and Ting Zhang and Yushuang Dong},
  title        = {Multilingual Encoder Knows more than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages},
  year         = {2025},
  url          = {http://dx.doi.org/10.13140/RG.2.2.11262.09285},
}
```

## 📄 License

This project is released under the MIT License. See the LICENSE file for details.
