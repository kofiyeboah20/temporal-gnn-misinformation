# Temporal Graph Neural Networks for Misinformation Detection

Code and experiment results for the MSc thesis:

**Modelling the Spread of Misinformation in Online Communities Using Temporal Graph Neural Networks**
Emmanuel Asiedu Yeboah — MSc Data Science, Department of Computer Science, University of Ghana (2026).
Supervisor: Dr. Alvin Kwabena Ansah.

## Overview

The study asks whether the timing of a misinformation cascade's early growth helps to tell
false from true beyond what its structure alone offers, and at what computational cost. A
Temporal Graph Attention Network (TGAT) is compared against logistic regression, a static
graph convolutional network, and an LSTM, under a leakage-controlled early-detection protocol
on the TAGFN dataset. The experiments cover detection at four observation windows (10%, 20%,
30%, 50%), an ablation of temporal information, a decomposition of the temporal signal into
speed, depth and breadth, and a scalability study.

## Repository structure

```
notebooks/
  TAGFN_pipeline.ipynb        End-to-end pipeline: data preparation, leakage check,
                              baselines, TGAT, ablation, decomposition, and scalability.
results/
  results_politifact.json     Per-seed metrics (accuracy, macro-F1, AUC) and ablation, PolitiFact.
  results_gossipcop.json      Per-seed metrics and ablation, GossipCop.
  scalability_fakeddit.json   Preparation time, peak memory, time per epoch and throughput,
                              on Fakeddit samples (1k to 200k cascades; OOM at 300k).
```

## Dataset

The experiments use the **TAGFN** dataset (Liu et al., 2026), publicly available at
<https://huggingface.co/datasets/kayzliu/TAGFN>. The three subsets used are PolitiFact and
GossipCop (detection and ablation) and Fakeddit (scalability only). The dataset is **not**
redistributed here; it belongs to its original authors and is downloaded directly from
Hugging Face by the notebook. The official train/validation/test splits distributed with the
dataset are used unchanged for every model.

## Reproducing the results

The notebook was developed and run on Google Colaboratory with a single NVIDIA T4 GPU (16 GB),
using PyTorch and PyTorch Geometric.

1. Open `notebooks/TAGFN_pipeline.ipynb` in Google Colab (or a local Jupyter environment with a
   CUDA GPU).
2. Run the cells in order. The notebook downloads TAGFN from Hugging Face, builds each cascade as
   a time-ordered event stream, verifies the masking guarantee, trains the baselines and the TGAT
   model over five random seeds (0–4), and runs the ablation, decomposition and scalability
   studies.
3. The metrics written to `results/` reproduce the tables and figures reported in Chapter Four of
   the thesis. Because GPU training is not bit-for-bit deterministic, exact values may differ
   slightly between runs; the reported figures are means over five seeds.

Key configuration (see Table 3.2 of the thesis): TGAT with 2 attention layers, 32 hidden units,
2 heads and a 16-dimensional time encoding; Adam at learning rate 0.005, weight decay 0.0001;
up to 300 epochs with early stopping on validation AUC (patience 40).

## Citation

> Asiedu Yeboah, E. (2026). *Modelling the Spread of Misinformation in Online Communities Using
> Temporal Graph Neural Networks* [MSc thesis]. University of Ghana.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The TAGFN dataset
is the property of its original authors and is subject to its own terms.
