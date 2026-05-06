# MicroGPT Extension

<div align="center">

### Building modern GPT architectures from scratch, one concept at a time.

A pedagogical extension of Andrej Karpathy’s minimalist GPT implementation, progressively augmented with modern transformer innovations.

Built to understand **how contemporary LLMs actually work under the hood** — not just use them.

---

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Educational](https://img.shields.io/badge/Purpose-Learning-green)
![Transformer](https://img.shields.io/badge/Focus-LLMs-orange)
![From Scratch](https://img.shields.io/badge/Implementation-From%20Scratch-red)

</div>

---

## Overview

This repository is my attempt at extending the beautifully minimal GPT implementation philosophy popularized by Andrej Karpathy's microGPT / nanoGPT-style educational codebases.

The original idea is simple:

> Strip away every engineering abstraction until only the core algorithm remains.

This project builds on that foundation by incrementally implementing many of the architectural ideas that power modern large language models.

The goal is not raw performance.

The goal is **understanding**.

Instead of treating transformer improvements as black-box tricks, each feature here is implemented directly from scratch to expose the mathematical and architectural intuition behind it.

---

## What This Repo Adds

This implementation extends the baseline GPT architecture with modern transformer mechanisms including:

### Tokenization

- Byte Pair Encoding (BPE)
- Custom vocabulary construction
- Byte-level tokenization pipeline

Why it matters:
Modern LLMs don't operate directly on characters or words. Understanding tokenization is foundational to understanding everything downstream.

---

### Attention Optimizations

#### Flash Attention

Memory-efficient exact attention computation.

Implemented to understand:

- IO-aware attention computation
- Numerical stability tricks
- Why naive attention becomes the bottleneck

---

#### Multi-Query Attention (MQA)

Reduces KV cache size by sharing keys and values across heads.

Useful for understanding inference optimization in production LLM systems.

---

#### Grouped Query Attention (GQA)

A middle ground between:

- Full Multi-Head Attention
- Multi-Query Attention

Used in many modern open-weight models because it offers strong quality-efficiency tradeoffs.

---

## Positional Encoding Variants

One of the most important and evolving areas of transformer design.

This repo explores multiple approaches.

---

### Rotary Positional Embeddings (RoPE)

Implements rotary transformations directly at the attention level.

Covers:

- Pairwise rotation intuition
- Complex-plane interpretation
- Relative position emergence

---

### ALiBi

Attention with Linear Biases.

Implemented to study:

- Position biasing without explicit embeddings
- Length extrapolation behavior
- Simplicity vs expressiveness tradeoffs

---

### T5 Relative Position Bias

Bucketed relative positional attention bias.

Useful for understanding:

- Relative position discretization
- Learned attention priors
- Why encoder-decoder architectures often prefer this style

---

## Architectural Experiments

The repository is designed as a playground for architectural exploration.

Examples include:

- Alternative attention formulations
- Efficiency-oriented transformer modifications
- Comparative positional encoding studies
- Ablation-friendly modular structure

---

## Why This Exists

Reading papers is useful.

Using HuggingFace is practical.

But implementing these ideas from scratch forces a very different level of understanding.

This repo was built as a way to answer questions like:

- What actually changes when switching from MHA to GQA?
- How does RoPE mathematically rotate embeddings?
- Why does Flash Attention save memory?
- What tradeoffs do relative positional methods introduce?

If you've ever read a transformer paper and thought:

> “I understand this… until I try implementing it.”

this repo is for that gap.

---

## Design Philosophy

This project follows three principles:

### 1. Minimal Dependencies

Keep implementation close to raw Python.

Less framework magic.

More visibility.

---

### 2. Explicit Math

No hidden abstractions.

Every tensor transformation should be understandable.

---

### 3. Learn by Rebuilding

The fastest way to deeply understand LLMs is to reconstruct them.

---

## Repository Structure

```text
microgpt_extension/
│
├── models/             # Modular model implementations
│   ├── __init__.py     # Model registry
│   ├── baseline.py     # Baseline with absolute positional embeddings
│   ├── rope.py         # Rotary Position Embeddings
│   ├── alibi.py        # Attention with Linear Biases
│   ├── t5_bias.py      # T5-style relative positional bias
│   ├── flash.py        # Flash Attention
│   └── xpos.py         # xPOS (extrapolation-friendly positions)
├── src.py              # Core algorithm and utilities (dataset, BPE, etc.)
├── utils.py            # Value class for autograd
├── train.py            # Training script with model selection
├── config.py           # Configuration
├── input.txt           # Dataset
└── resources/          # Related papers
```

---

## Using Different Models

The codebase is organized as follows:

Each model variant is now in its own file under the `models/` directory for better readability and modularity. The `models/__init__.py` file contains a `MODEL_REGISTRY` that maps model names to their implementations.

To train or experiment with a specific model architecture, simply use the `--model` flag when running `train.py`.

---

## Implemented Concepts

| Category | Features |
|---------|----------|
| Tokenization | BPE |
| Attention | Standard MHA, Flash Attention, MQA, GQA |
| Positional Encoding | Learned, RoPE, ALiBi, T5 Bias, xPOS |
| Training | GPT training loop with model selection |
| Inference | Autoregressive decoding |

---

## Inspiration

This project draws inspiration from:

- Andrej Karpathy’s educational GPT implementations
- nanoGPT
- minBPE
- Flash Attention papers
- RoPE / ALiBi / T5 architecture work
- Modern open-weight LLM design choices

---

## Why This Is Probably Useful

Most educational GPT repos stop at:

- vanilla self-attention
- learned positional embeddings
- simple tokenization

This repo pushes further into concepts that actually matter in modern LLM design.

It tries to bridge the gap between:

**toy transformer implementations**  
and  
**real architectural ideas used in production models**

---

## Future Extensions

Potential additions:

- Mixture of Experts
- Sliding Window Attention
- Speculative Decoding
- State Space Models
- YaRN / NTK RoPE scaling
- Multi-token prediction
- Prefix LM masking

---

## Running

Train a model with a specific architecture:

```bash
# Train with baseline model (default)
python3 train.py

# Train with RoPE positional embeddings
python3 train.py --model rope

# Train with ALiBi
python3 train.py --model alibi

# Train with T5 relative positional bias
python3 train.py --model t5_bias

# Train with Flash Attention
python3 train.py --model flash

# Train with xPOS
python3 train.py --model xpos

# Customize training steps and samples
python3 train.py --model rope --num-steps 1000 --num-samples 50
```

### Command Line Arguments

- `--model {baseline,rope,alibi,t5_bias,flash,xpos}`: Choose model architecture (default: baseline)
- `--num-steps`: Number of training steps (default: 500)
- `--num-samples`: Number of inference samples to generate (default: 20)

---

## Acknowledgements

Huge credit to Andrej Karpathy for demonstrating that complex systems become understandable when stripped to their essence.

This repository exists because educational codebases like microGPT make deep learning implementation approachable.

---

## Final Note

This is not meant to compete with optimized frameworks.

It is meant to make transformer internals tangible.

If this repo helps someone finally *understand* why these architectural ideas exist, it has done its job.
