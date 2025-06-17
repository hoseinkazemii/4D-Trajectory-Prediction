# Generalizable Deep Sequence Models for 4D Trajectory Prediction of Tower Crane Loads

This repository accompanies the paper:

> **Kazemi *et al.*** “Generalizable Deep Sequence Models for 4D Trajectory Prediction of Tower Crane Loads”  
> DOI:&nbsp;`10.XXXX/zenodo.XXXXXXX` <!-- update once assigned -->

The project provides an **end-to-end framework** for forecasting the future Cartesian trajectory (X&nbsp;Y&nbsp;Z + time) of suspended loads moved by tower cranes.  

---

## ✨ Key contributions

| Aspect | Highlights |
|--------|------------|
| **Data generation** | Unity-based digital twin that emulates industrial rotary & linear encoders (θ, r, h) and logs time-synchronised crane states. |
| **Human-in-the-loop realism** | 29 participants executed randomized pick-and-place tasks, yielding a diverse dataset of operator-induced motion patterns. |
| **Model families** | 1) **Seq2Seq + Temporal Attention** • 2) **ConvLSTM** • 3) **Temporal Convolutional Network (TCN)** |
| **Robust evaluation** | Benchmarks cover spatial generalization to unseen logistics layouts, varying look-ahead horizons, sensor-noise injection, and lower sampling frequencies. |
| **Deployment-ready code** | Modular PyTorch/TensorFlow pipeline with configuration files, training & evaluation scripts, and a real-time inference demo. |

> **Results are intentionally omitted here** while the manuscript undergoes peer review.

---

## 📂 Repository layout

