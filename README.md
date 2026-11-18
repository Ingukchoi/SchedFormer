<h1 align="center">
Towards a Unified Model for 
  
Flexible Job Shop Scheduling with Diverse Constraints
</h1>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white"></a>
  <a href="https://pytorch-geometric.readthedocs.io/"><img src="https://img.shields.io/badge/PyTorch%20Geometric-6A0DAD?logo=pytorch&logoColor=white"></a>
  <a href="https://openreview.net/forum?id=sE50rZRh2S"><img src="https://img.shields.io/badge/NeurIPS%202026-Paper-B31B1B"></a>
</p>

>This repository is the official implementation of our paper, "**Towards a Unified Model for Flexible Job Shop Scheduling with Diverse Constraints**", The Fortieth Annual Conference on Neural Information Processing Systems (**NeurIPS 2026**).
>The code builds upon [DAN](https://github.com/wrqccc/fjsp-drl) repository.

<p align="center">
  <img src="./framework.png" width="700">
</p>

## Abstract

Deep learning-based heuristics have shown great promise for solving the Flexible Job Shop Scheduling Problem (FJSSP). However, existing studies typically overlook operational constraints prevalent in real-world industries or target only a specific FJSSP variant, limiting their practical applicability. In this paper, we introduce SchedFormer, a multi-task scheduling agent that solves FJSSP variants with diverse constraints and generalizes zero-shot to those unseen during training. To support a range of variants within a common framework, we propose a Unified State Representation (USR) that maps different types of constraints into a shared feature space encoding their combined temporal effects, and captures invariant operation-machine relationships across variants via four semantically distinct graphs. To fully exploit the structure of the USR, we also develop a unified neural solver with graph-specific Transformer blocks.
Each block employs an attention mechanism specialized for its target graph, while the mapped constraint information conditions the overall representation learning. Experiments on 16 FJSSP variants show that SchedFormer compares favorably with baselines, including specialized neural solvers, and generalizes very well to unseen variants. Further experiments demonstrate its applicability to other scheduling problems.

## FJSSP variants

Any subset can be combined, giving **16 FJSSP variants**.

| FJSSP variants | Classic FJSSP | Release dates<br>(R) | Sequence-dependent setup times<br>(S) | Machine maintenance<br>(M) | Transportation times<br>(T) |
|:---------------|:-------------:|:--------------------:|:------------------:|:--------------------------:|:---------------------------:|
| **FJSSP**      | ✓ |   |   |   |   |
| **FJSSP-R**    | ✓ | ✓ |   |   |   |
| **FJSSP-S**    | ✓ |   | ✓ |   |   |
| **FJSSP-M**    | ✓ |   |   | ✓ |   |
| **FJSSP-T**    | ✓ |   |   |   | ✓ |
| **FJSSP-RT**   | ✓ | ✓ |   |   | ✓ |
| **FJSSP-RS**   | ✓ | ✓ | ✓ |   |   |
| **FJSSP-RM**   | ✓ | ✓ |   | ✓ |   |
| **FJSSP-SM**   | ✓ |   | ✓ | ✓ |   |
| **FJSSP-ST**   | ✓ |   | ✓ |   | ✓ |
| **FJSSP-MT**   | ✓ |   |   | ✓ | ✓ |
| **FJSSP-RSM**  | ✓ | ✓ | ✓ | ✓ |   |
| **FJSSP-RST**  | ✓ | ✓ | ✓ |   | ✓ |
| **FJSSP-RMT**  | ✓ | ✓ |   | ✓ | ✓ |
| **FJSSP-SMT**  | ✓ |   | ✓ | ✓ | ✓ |
| **FJSSP-RSMT** | ✓ | ✓ | ✓ | ✓ | ✓ |

## Quick Start

### ⚙️ **Requirements**
- `python=3.10.15`
- `torch==2.2.0`
- `torch_geometric==2.5.2`
- `numpy==1.26.4`
- `tqdm==4.66.5`

---

### 🗂️ Repository layout

```
SchedFormer/
├── train.py              # training (Adaptive self-improvement learning)
├── test.py               # inference (greedy / sampling)
├── test_heuristic.py     # PDRs
├── data_utils.py         # instance generator + dataset writer
├── SFEnv.py              # Unified FJSSP simulator (MultiFJSSPEnv)
├── common_utils.py       # graph construction, action selection, heuristics, seeding
├── params.py             # all command-line arguments
├── model/
│   ├── SchedFormer.py    # model network
│   ├── attention_layer.py# attention blocks
│   └── sub_layers.py     # MLP, RMSNorm, SwiGLU
├── data/Synthetic/       # generated test instances, {variant}/{size}/*.npz
└── train_results/        # trained checkpoints, {run}/FJSSP/models/checkpoint{n}.pth
```

### 💾 Data

The benchmark datasets used in our experiments are available in
[GitHub Releases](https://github.com/Ingukchoi/SchedFormer/releases/tag/dataset-v1.0).
The datasets contain 16 problem variants × 3 instance sizes × 100 instances
(≈360 MB compressed, ≈2.8 GB uncompressed).
Download and extract the datasets into the `data/` directory.
Test instances follow the directory structure:
`data/Synthetic/{variant}/{n_j}x{n_m}/*.npz`.

---

### 🏃 **Train**

```bash
# single problem variant (here: all four constraints active)
python train.py --run_name SF_RSMT \
                --flag_ready True --flag_setup True --flag_maintenance True --flag_trans True \
                --n_j 10 --n_m 5

# multi-task over the 4 single-constraint variants
python train.py --run_name SF_4V --flag_multi True --flag_full False

# multi-task over all 16 variants
python train.py --run_name SF_16V --flag_multi True --flag_full True
```

---

### 🏃‍♂️ **Test**

```bash
# greedy decoding
python test.py --test_model SF_16V --epoch 4000 --test_problem FJSSP-RSMT --test_size 10x5

# sampling: draw 100 schedules per instance, keep the best
python test.py --test_model SF_16V --epoch 4000 --test_problem FJSSP-RSMT --test_size 10x5 \
               --test_mode True --sample_times 100
```
---
📝 If you find our code valuable for your research, please cite:
```bibtex
@inproceedings{
choi2026towards,
title={Towards a Unified Model for Flexible Job Shop Scheduling with Diverse Constraints},
author={Choi, Inguk and Shin, Woo-Jin and Cho, Sang-Hyun and Kim, Hyun-Jung},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026}
}
```
