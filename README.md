# shakespeareGPT

A GPT-2 style transformer trained from scratch on Shakespeare text, built with PyTorch.

## Overview

This project implements a 124M parameter GPT model architecture and trains it on Shakespeare's works to generate Shakespearean text. The model is built from the ground up — including multi-head attention, transformer blocks, layer norm, and GELU activations.

## Architecture

- **Model**: GPT-2 style decoder-only transformer (124M params)
- **Vocab size**: 50,257 (GPT-2 BPE tokenizer via `tiktoken`)
- **Context length**: 256 tokens
- **Embedding dim**: 768
- **Attention heads**: 12
- **Layers**: 12
- **Dropout**: 0.1

## Project Structure

```
shakespeareGPT/
├── main.py          # Training entrypoint
├── architecture.py  # GPTModel, TransformerBlock, MultiHeadAttention, etc.
├── data_prep.py     # Dataset and dataloader (sliding window tokenization)
├── util.py          # Training loop, loss calculation, text generation helpers
├── training_text/   # Raw text data (Shakespeare)
└── model.pth        # Saved model weights
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA-capable GPU.

## Training

```bash
python main.py
```

Training config:
- Optimizer: AdamW (`lr=0.0004`, `weight_decay=0.1`)
- Batch size: 2
- Epochs: 10
- 90/10 train/val split

Weights are saved to `model.pth` after training.

## Text Generation

After training, the model generates text from a prompt:

```
Input:  "Thou art the"
Output: [generated Shakespeare-style continuation]
```

Generation uses greedy decoding (argmax over softmax logits).