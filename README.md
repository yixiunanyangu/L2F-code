<div align="center">

# 🖥️ L2F: Learning to Focus on Logical Regions for High-Resolution GUI Grounding

<p align="center">
  <img src="https://img.shields.io/badge/Task-GUI%20Grounding-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Base%20Model-Qwen2.5--VL-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-13k%20Samples-blue?style=for-the-badge"/>
</p>

<p align="center">
  📄 <a href="https://yixiunanyangu.github.io/l2f-demo/"><strong>Project Page</strong></a> &nbsp;|&nbsp;
  🤗 <a href="https://yixiunanyangu.github.io/l2f-demo/"><strong>Dataset (Coming Soon)</strong></a> &nbsp;|&nbsp;
  💻 <a href="https://github.com/yixiunanyangu/L2F-code"><strong>Code</strong></a>
</p>

</div>

---

## 📖 Abstract

GUI grounding remains challenging, particularly in high-resolution displays with small icons and cluttered interfaces. We propose **L2F**, a framework that enables models to learn to focus on semantically coherent **logical regions** — mimicking how humans first identify a functional context (e.g., a specific window or panel) before locating the precise target element.

We contribute:
- 🗂️ **L2F Dataset** — 13k expert-annotated instruction–screenshot pairs with logical region labels across three annotation levels (WLR / WSLR / ELR)
- 🔌 **L2F-PLR** — A plug-in logical region detection model compatible with any existing GUI grounding model
- 🔄 **L2F-E2E** — An end-to-end model that jointly reasons over logical regions and target elements in a single pass

> **L2F-PLR** achieves an average absolute gain of **+12.95%** across existing models. **L2F-E2E-7B** achieves **50.86%** overall accuracy on ScreenSpot-Pro, surpassing all prior state-of-the-art methods.

---

## 🏆 Main Results

### ScreenSpot-Pro (Overall Accuracy %)

| Model | Text | Icon | **Avg** |
|:---|:---:|:---:|:---:|
| Qwen2.5-VL-3B | 22.72 | 3.31 | 15.31 |
| + **L2F-PLR** | 41.04 | 9.77 | **29.09** ↑+13.78 |
| Qwen2.5-VL-7B | 39.09 | 6.95 | 26.82 |
| + **L2F-PLR** | 52.30 | 12.75 | **37.19** ↑+10.37 |
| CogAgent-24 | 17.29 | 5.29 | 12.71 |
| + **L2F-PLR** | 48.62 | 11.75 | **34.53** ↑+**21.82** |
| GUI-Actor-3B | 57.98 | 18.24 | 42.20 |
| + **L2F-PLR** | 63.15 | 26.16 | **49.02** ↑+6.82 |
| SE-GUI-7B | 63.50 | 21.00 | 47.30 |
| OpenCUA-7B | — | — | 50.00 |
| **L2F-E2E-3B** | 54.45 | 13.74 | **38.90** |
| **L2F-E2E-7B** | **72.65** | 15.14 | **50.86** 🏆 |

### Generalization on Other Benchmarks

| Model | L2F-test | AITW | Mind2Web | ScreenSpot-v2 | ScreenSpot |
|:---|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-VL-3B | 29.45 | 37.94 | 17.68 | 70.52 | 72.56 |
| + **L2F-PLR** | 37.00 ↑+7.55 | 40.25 ↑+2.31 | 24.55 ↑+6.87 | 79.17 ↑+8.65 | 75.62 ↑+8.11 |
| **L2F-E2E-3B** | 45.36 | 46.27 | 26.63 | 84.20 | 81.92 |
| **L2F-E2E-7B** | **62.82** | **52.53** | **34.86** | **91.87** | **89.79** |

---

## ⚙️ Getting Started

### 1. Install Dependencies

```bash
pip install -r environment/vlm-r1-requirements.txt
```

### 2. Download Base Models

- [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

Other models referenced in evaluation (SeeClick, uGround, ShowUI, etc.) should be downloaded from their respective repositories.

### 3. Train

```bash
bash src/open-r1-multimodal/run_scripts/train.sh
```

### 4. Evaluate

```bash
# Evaluate with Qwen2.5-VL-3B
bash evalue/run_Qwen3B.sh
```

For other model variants, refer to the corresponding scripts in the `evalue/` directory.

---

## 📝 Citation

```bibtex
@article{l2f2026,
  title  = {L2F: Learning to Focus on Logical Regions for High-Resolution GUI Grounding},
  year   = {2026}
}
```
