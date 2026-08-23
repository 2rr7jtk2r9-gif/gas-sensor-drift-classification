# Gas Sensor Drift Classification

Time-series classification of e-nose gas sensor signals using SVM, LSTM, Transformer 
and Echo State Network (ESN) models, with a focus on long-term sensor drift robustness 
and TinyML feasibility for embedded deployment.

## Overview

Electronic nose (e-nose) systems used in industrial gas monitoring suffer from **sensor 
drift** — sensor responses degrade over time, causing models trained on early data to 
lose accuracy on later data. This project benchmarks four modelling approaches on the 
same dataset and evaluation protocol to find which one is most robust to drift, and 
whether it's feasible to deploy on low-power embedded hardware.

## Dataset

UCI Gas Sensor Array Drift Dataset (Vergara et al., 2012)
https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

- 13,910 samples, 128 features, 6 gas classes, collected over 36 months across 10 batches

## Methods

- **SVM** — RBF kernel, z-score normalisation, grid search, 10-fold CV
- **LSTM** — 2-layer, 64 hidden units, Adam optimiser
- **Transformer** — TransformerEncoderLayer, d_model=128, nhead=8
- **ESN** — Echo State Network (reservoir computing) via ReservoirPy

Each model is evaluated two ways:
1. **Mixed-split** — standard stratified train/test split
2. **Drift experiment** — trained on early batches (1–3), tested on late batches (8–10)

## Key Results

| Model | Mixed-split accuracy | Late-batch accuracy (drift) |
|---|---|---|
| SVM | 99.35% | 61.18% |
| Transformer | 98.94% | 55.62% |
| ESN | — | 54.59% |
| LSTM | 98.67% | 43.88% |

SVM had the best raw drift accuracy; ESN degraded the least between best and worst batch.

## TinyML Feasibility

SVM and LSTM models were exported to ONNX and benchmarked for size and CPU inference 
latency, to assess viability for real-time gas monitoring on embedded hardware.

## Repository Contents

- `gas-sensor-drift-classification.ipynb` — full notebook (data loading, EDA, all four 
  models, drift experiments, TinyML export)

## Author

Maharshi Vyas — MSc Advanced Electrical and Electronic Engineering, University of Leicester
