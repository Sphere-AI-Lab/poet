# POET & POET-X for LLM Pretraining

**Reparameterized LLM Training via Orthogonal Equivalence Transformation**

[![Paper](https://img.shields.io/badge/arXiv-2506.08001-b31b1b.svg)](https://arxiv.org/abs/2506.08001)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-4b44ce.svg)](https://neurips.cc/virtual/2025/poster/118691)
[![POET Page](https://img.shields.io/badge/Project-POET-orange)](https://spherelab.ai/poet/)

**POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation**

[![Paper](https://img.shields.io/badge/arXiv-2603.05500-b31b1b.svg)](https://arxiv.org/abs/2603.05500)
[![POET-X Page](https://img.shields.io/badge/Project-POET--X-blue)](https://spherelab.ai/poetx/)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [POET](#poet)
  - [Method](#method)
  - [Spectral Diversity](#spectral-diversity)
  - [Energy & Spectrum Preservation](#energy--spectrum-preservation)
  - [Efficient Approximation: SPO](#efficient-approximation-stochastic-primitive-optimization-spo)
  - [Results](#results)
- [POET-X](#poet-x)
  - [Overview](#overview-1)
  - [Key Results](#key-results)
  - [Pretraining Results](#pretraining-results)
  - [Memory Efficiency](#memory-efficiency)
  - [Throughput & Distributed Scaling](#throughput--distributed-scaling)
  - [Method: Key Optimizations](#method-key-optimizations)
  - [POET-X Variants](#poet-x-variants)
- [Citation](#citation)
- [Related Work](#related-work)

---

## Overview

This repository contains the official implementation of **POET** and **POET-X** — a family of reparameterized LLM training algorithms that optimize weight matrices through **Orthogonal Equivalence Transformation (OET)**, achieving superior generalization with provably bounded weight spectra.

<p align="center">
  <img src="assets/poet/teaser_poet.png" alt="POET three learning phases" width="85%">
  <br><em>POET's three learning phases: conical shell searching → stable learning → final adjusting.</em>
</p>

**Since POEX-X is an efficient version of POET without any approximation involved. Our repo will only provide the implementation of POET-X. The original implementation of POET is obsolete.**

---

## Installation

```bash
git clone https://github.com/Sphere-AI-Lab/poet.git
cd poet
pip install -e .
```

**Requirements:**
- Python ≥ 3.10
- PyTorch ≥ 2.7
- CUDA ≥ 12.6
- Triton ≥ 3.4.0

---

<!-- ## Data Preparation

The training scripts expect the C4 dataset at `./c4/en/` relative to the repo root. Run the following commands **from the root of this repository**:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/*"
cd ..
```

This will create a `c4/en/` folder directly inside the repo root, resulting in the following structure:

```
poet/
├── c4/
│   └── en/
│       ├── c4-train.00000-of-01024.json.gz
│       └── ...
├── torchrun_main.py
└── ...
```

> **Note:** If no local data is found, the training script will automatically fall back to streaming the dataset directly from HuggingFace (`allenai/c4`), which requires an internet connection but no local storage.


## Usage

### Quick Start

```bash
# Pretrain LLaMA-3B with POET-X (block_size=512) on C4
bash scripts/benchmark_c4_poet/pretrain_poet_3b.sh

# Pretrain LLaMA-3B with POET-XQ (block_size=512) on C4
bash scripts/benchmark_c4_qpoet/pretrain_qpoet_3b.sh
``` -->

<!-- ### Using POET/POET-X for LLM Pretraining

```python
from poet import POETConfig, wrap_model_with_poet, POETAdamW

config = POETConfig(
    variant="block_stochastic",  # "block_stochastic" (POET-BS) or "fully_stochastic" (POET-FS)
    block_size=128,              # Sampling budget b
    neumann_terms=5,             # Cayley-Neumann approximation order
    merge_interval=100,          # Steps between merge-then-reinitialize
)

model = ...  # Your LLaMA / transformer model
model = wrap_model_with_poet(model, config)

# Works with standard DDP (no FSDP needed)
model = torch.nn.parallel.DistributedDataParallel(model)

optimizer = POETAdam(model.parameters(), lr=1e-3)
```

### Merge Weights for Inference

After training, orthogonal matrices merge into base weights — **zero inference overhead**:

```python
from poet import merge_poet_weights

model = merge_poet_weights(model)  # W ← R W_0 P
model.save_pretrained("./my-pretrained-llm")
``` -->

## Quick Start

Get started with POET in just a few lines of code:

```python
from poet_torch import POETConfig, POETModel, get_poet_optimizer

# 1. Configure POET
config = POETConfig(
    block_size=256,       # Block size for block-stochastic optimization
    merge_interval=200,   # Steps between merge-then-reinitialize
)

# 2. Wrap your model with POET
model = POETModel(your_model, config)

# 3. Create optimizer (automatically handles orthogonal parameters)
optimizer = get_poet_optimizer(model, lr=1e-3)

# 4. Training loop
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    loss = model(**batch)
    loss.backward()
    optimizer.step()
    model.merge_if_needed(step)  # Periodic merge (zero overhead after training)
```

### Key Components

| Component | Description |
|-----------|-------------|
| `POETConfig` | Configuration for block size, merge interval, and variant selection |
| `POETModel` | Wraps your model to apply Orthogonal Equivalence Transformation |
| `get_poet_optimizer()` | Creates an optimizer tailored for POET's orthogonal parameters |
| `merge_if_needed()` | Periodically absorbs orthogonal matrices into base weights |

**📁 More Examples:** Explore comprehensive training scripts and additional APIs in the [`examples/`](examples/) directory. 

## POET

### Method

POET reparameterizes each weight matrix as:

$$W_{RP} = R \, W_0 \, P$$

where $W_0 \in \mathbb{R}^{m \times n}$ is a **fixed** randomly initialized matrix, and $R \in \mathbb{R}^{m \times m}$, $P \in \mathbb{R}^{n \times n}$ are **learnable orthogonal matrices**. Training only updates $R$ and $P$, leaving $W_0$ unchanged.

**Why orthogonal transformations?** They preserve singular values exactly — giving POET direct, provable control over the weight spectrum throughout training.

<p align="center">
  <img src="assets/poet/sv_comp.png" alt="Singular value dynamics" width="55%">
  <br><em>Dynamics of singular values: POET (right) avoids the large singular value growth seen in standard AdamW training (left).</em>
</p>

### Spectral Diversity

<p align="center">
  <img src="assets/poet/svd_entropy.png" alt="SVD entropy comparison" width="90%">
  <br><em>POET maintains consistently higher SVD entropy (singular value diversity) throughout training compared to AdamW and Muon.</em>
</p>

### Efficient Approximation: Stochastic Primitive Optimization (SPO)

Large orthogonal matrices $R \in \mathbb{R}^{m \times m}$ are expensive to optimize naively. POET introduces two efficient variants:

- **POET-FS** (Fully Stochastic SPO): Randomly samples a small $b \times b$ submatrix at each step. Highly parameter-efficient; decouples parameter count from matrix size.
- **POET-BS** (Block-Stochastic SPO): Block-diagonal structure with random permutations; transforms all dimensions simultaneously. More expressive per parameter.

<p align="center">
  <img src="assets/poet/block_pattern.png" alt="Weight update patterns" width="85%">
  <br><em>Weight update coverage: POET-BS achieves more even updates across all weight elements compared to POET-FS.</em>
</p>

Orthogonal matrices are parameterized via **Cayley-Neumann Parameterization (CNP)**, which approximates the matrix inverse using a truncated Neumann series for numerical stability:

$$R = (I + Q)(I - Q)^{-1} \approx (I + Q)\left(I + \sum_{i=1}^{k} Q^i\right)$$

A **merge-then-reinitialize** trick periodically absorbs $R, P$ into $W_0$, preventing error accumulation and keeping the Neumann series convergent.

### Results

<p align="center">
  <img src="assets/poet/val_ppl.png" alt="Validation perplexity vs parameters" width="60%">
  <br><em>POET outperforms AdamW with significantly fewer trainable parameters across all LLaMA model sizes on C4.</em>
</p>

<div align="center">
  
| Method | Params | 60M PPL | 130M PPL | 350M PPL | 1.3B PPL |
|---|---|---|---|---|---|
| AdamW | Full | 26.68 | 20.82 | 16.78 | 14.73 |
| GaLore | Full | 29.81 | 22.35 | 17.99 | 18.33 |
| LoRA (r=64) | ~5% | 39.70 | 32.07 | 25.19 | 20.55 |
| POET-BS (b=128) | ~13% | **26.90** | **21.86** | **18.05** | **16.24** |
| POET-BS (b=256) | ~26% | **25.29** | **19.88** | **16.27** | **14.56** |

<em>Quantitative comparison of validation perplexity</em>
</div>

<p align="center">
  <img src="assets/poet/ablation_training_speedup.png" alt="Training speedup" width="35%">
  <br><em>POET-FS (b=1/2) still outperforms AdamW even when AdamW is trained with ~3× more tokens.</em>
</p>


## POET-X

### Overview

POET-X is a **scalable, memory-efficient** variant of POET that makes orthogonal equivalence training practical at the billion-parameter scale.

> The original POET must store the full transformed weight $RW_0P$ for backpropagation, making it **more memory-intensive than AdamW**. POET-X resolves this through a suite of engineering innovations.

### Key Results

<p align="center">
  <img src="assets/poetx/latency_breakdown.png" alt="Latency breakdown" width="55%">
  <br><em>Latency breakdown: POET-X reduces forward+backward latency from 10.59ms (POET) to 1.38ms (POET-Xfast), approaching standard linear layers.</em>
</p>

<p align="center">
  <img src="assets/poetx/mem_breakdown.png" alt="Memory breakdown" width="55%">
  <br><em>Memory breakdown for Llama-8B training on a single GPU. POET-X_mem achieves PEFT-level memory; POET runs OOM.</em>
</p>

### Pretraining Results

<p align="center">
  <img src="assets/poetx/ppl_results.png" alt="PPL results" width="45%">
  <br><em>Llama-3B pretraining on 60B C4 tokens: POET-X achieves better PPL than AdamW and all memory-efficient baselines.</em>
</p>

<p align="center">
  <img src="assets/poetx/ppl_results_q.png" alt="PPL results quantized" width="45%">
  <br><em>POET-XQ (quantized): Best PPL of 14.78 with minimal memory footprint, outperforming GaLore and APOLLO.</em>
</p>

Training dynamics with different block sizes:

<p align="center">
  <img src="assets/poetx/val_ppl_256.png" alt="Val PPL b=256" width="30%">
  <img src="assets/poetx/val_ppl_1024.png" alt="Val PPL b=1024" width="30%">
  <br><em>Validation PPL curves at block size b=256 (left) and b=1024 (right).</em>
</p>

### Memory Efficiency

<p align="center">
  <img src="assets/poetx/peak_mem.png" alt="Peak GPU memory" width="80%">
  <br><em>Peak GPU memory across model sizes (3B–13B) and sequence lengths: POET-X_mem outperforms all baselines including LoRA.</em>
</p>

### Throughput & Distributed Scaling

<p align="center">
  <img src="assets/poetx/throughput_scale.png" alt="Throughput scaling" width="75%">
  <br><em>POET-X closely follows ideal linear scaling on 64× H100s, while AdamW (FSDP) plateaus due to communication overhead.</em>
</p>

### Method: Key Optimizations

The core insight is an **input-centric formulation** that avoids materializing the full $m \times n$ transformed weight:

$$z = \underbrace{\Phi_n G_P^\top \Phi_n^\top}_{P^\top} W \underbrace{\Phi_m G_R^\top \Phi_m^\top}_{R^\top} x$$

This reduces complexity from $O(nm^2)$ to a sequence of matrix-vector products.

Four engineering innovations:

1. **Permutation Acceleration** — Custom CUDA kernels for index-mapped permutations (up to **20× speedup**).
2. **Permutation Reduction** — Pre-computes permuted weights once per inner loop, eliminating redundant ops.
3. **Batch-Parallel Strategy** — Treats each block of block-diagonal $G_P$, $G_R$ as an independent batch element; avoids large sparse matrix construction.
4. **Fused Cayley-Neumann Kernels** — Triton kernel loads $Q$ and $Q^2$ into shared memory once for all terms; backward pass also fused.

<p align="center">
  <img src="assets/poetx/cayley.png" alt="Cayley-Neumann illustration" width="45%">
  <br><em>Fused Cayley-Neumann parameterization: batch-wise implementation via Triton kernel fusion.</em>
</p>

### POET-X Variants

<div align="center">
  
| Variant | Memory | Speed | Notes |
|---|---|---|---|
| `POET-X_fast` | Medium | Fast | Standard autograd, saves activation $b$ |
| `POET-X_mem` | **Lowest** | Moderate | Gradient checkpointing, recomputes $b$ on-the-fly |
| `POET-XQ` | **Lowest** | High throughput | INT8 quantized base weights, dequantized on-the-fly |

</div>

---

## Citation

```bibtex
@article{qiu2025poet,
  title={Reparameterized LLM Training via Orthogonal Equivalence Transformation},
  author={Qiu, Zeju and Buchholz, Simon and Xiao, Tim Z. and Dax, Maximilian and Sch{\"o}lkopf, Bernhard and Liu, Weiyang},
  journal={arXiv preprint arXiv:2506.08001},
  year={2025}
}

@article{qiu2025poetx,
  title={POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation}, 
  author={Qiu, Zeju and Liu, Lixin and Weller, Adrian and Shi, Han and Liu, Weiyang},
  journal={arXiv preprint arXiv:2603.05500},
  year={2026},
}
```

---

## Related Work

- [OFT](https://github.com/Zeju1997/oft) — Orthogonal Finetuning for diffusion models  
- [GaLore](https://github.com/jiaweizzhao/GaLore) — Gradient low-rank projection  
- [Muon](https://github.com/KellerJordan/Muon) — Gradient orthogonalization optimizer  
