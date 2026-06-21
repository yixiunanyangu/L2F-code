<div align="center">

# 🖥️ L2F: Learning to Focus on Logical Regions for High-Resolution GUI Grounding

<p align="center">
  <img src="https://img.shields.io/badge/Task-GUI%20Grounding-green?style=for-the-badge" alt="GUI Grounding"/>
  <img src="https://img.shields.io/badge/Base%20Model-Qwen2.5--VL-orange?style=for-the-badge" alt="Qwen2.5-VL"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
</p>

</div>

---

## 📖 Abstract

Applying Multi-modal Large Language Models (MLLMs) to develop GUI agents for daily applications such as online shopping and web browsing has achieved significant progress. However, **GUI grounding remains a challenging task**, particularly in scenarios involving high-resolution displays and small icon sizes.

In this paper, we propose enabling the agent to **learn to focus on target regions** imitating the logical perceptual patterns of humans. Specifically, we:

- 🗂️ Construct a **logical region dataset named L2F** with 13,000 expert-annotated instruction–screenshot pairs, annotating boundary boxes containing grounding elements mimicking human attention mechanisms.
- 🔌 Introduce a **light plug-in detection model (P-LR)** that can be seamlessly integrated with any existing grounding model.
- 🔄 Present an **end-to-end model (E2E-GG)** that directly generates final outputs using logical regions as intermediate reflection to guide reasoning.

> **Key Results:** P-LR improves existing models by an average absolute gain of **+12.95%**, and E2E-GG achieves **50.86%** overall accuracy on ScreenSpot-Pro, surpassing prior state-of-the-art methods.

---

## 🔍 Motivation

<div align="center">

| Traditional Local Cropping ❌ | Our Logical Region Approach ✅ |
|:---:|:---:|
| Crops based on pixel proximity | Crops based on semantic coherence |
| Contains multiple visual distractors | Encapsulates functionally related elements |
| Treats GUI as a flat pixel grid | Models functional structure of interfaces |

</div>

Humans naturally resolve GUI ambiguity by first identifying the relevant **functional context** (e.g., a specific application window or control panel), before locating the precise element. Our **logical region** paradigm mirrors this cognitive process.

---

## 🏗️ Framework Overview

We propose two complementary model architectures trained with **two-stage GRPO**:

### 🔌 Plug-in Logical Region Detection Model (P-LR)
- Preprocesses input screenshots by predicting and cropping to the logical region
- Provides downstream MLLMs with a cleaner, context-aware visual input
- **Compatible with any existing GUI grounding model**
- Trained with a multi-dimensional reward mechanism: Logic-Consistency (*R*ℓ), Regional-Consistency (*R*_iou), Regional-Constraint (*R*_C), and Format Reward (*R*_F)

### 🔄 End-to-End GUI Grounding Model (E2E-GG)
- Unifies logical reasoning and target localization into a **single pipeline**
- Follows "first locate the logical region, then pinpoint the task target" paradigm
- Features a **Dynamic Weighted Reward Function (DWRF)** that progressively increases focus on task-point accuracy during training
- Eliminates intermediate dependencies while preserving logical awareness

---

## 📊 Dataset: L2F

| Property | Details |
|:---|:---|
| **Scale** | 13,000 expert-annotated instruction–screenshot pairs |
| **Resolution** | Min 1,280×720; majority at 2,880×1,800; up to 3,360×2,100 |
| **Sources** | ShowUI, OS-ATLAS, UI-Vision, Aria-UI (~60k raw samples) |
| **Annotation Region Size** | Fixed range: 600×600 to 1,024×1,024 pixels |

### Three-Level Logical Region Annotation Strategy

```
📐 Window-Level Logical Region (WLR)
   └─ Multi-window scenarios: identifies the specific window containing the task

📐 Workspace-Level Logical Region (WSLR)
   └─ Single-interface: segments functional areas (toolbars, panels, etc.)

📐 Element-Level Logical Region (ELR)
   └─ Simple interfaces: centers on task-related elements
```

### Five Application Domains
- 🖥️ System Platforms
- 🏢 Professional & Productivity Applications
- 📱 Daily Utilities & Information Access
- ⚙️ System Operations & Management
- 🌐 Network & Entertainment Interaction

---

## 📈 Main Results

### ScreenSpot-Pro Benchmark (Overall Accuracy %)

| Model | Text | Icon | **Avg** |
|:---|:---:|:---:|:---:|
| Qwen2.5-VL-3B | 22.72 | 3.31 | 15.31 |
| Qwen2.5-VL-3B + **P-LR** | 41.04 | 9.77 | **29.09** ↑+13.78 |
| Qwen2.5-VL-7B | 39.09 | 6.95 | 26.82 |
| Qwen2.5-VL-7B + **P-LR** | 52.30 | 12.75 | **37.19** ↑+10.37 |
| GUI-Actor-3B | 57.98 | 18.24 | 42.20 |
| GUI-Actor-3B + **P-LR** | 63.15 | 26.16 | **49.02** ↑+6.82 |
| **E2E-GG-3B** | 54.45 | 13.74 | **38.90** |
| **E2E-GG-7B** | 72.65 | 15.14 | **50.86** 🏆 |

> **P-LR achieves up to +21.82% absolute improvement** (CogAgent-24), outperforming the GUI-Actor+Verifier variant by an average of 3.05%.

### Cross-Dataset Generalization

P-LR and E2E-GG also demonstrate strong generalization on:
- **ScreenSpot-Pro** — high-resolution professional GUI benchmark
- **L2F-test** — 1,100 manually annotated custom test samples
- **AITW** — Android in the Wild dataset
- **Mind2Web** — web-oriented grounding tasks

---

## ⚙️ Getting Started

### Prerequisites

**1. Install dependencies**

```bash
pip install -r environment/vlm-r1-requirements.txt
```

**2. Download the required base models:**

- [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

Other models referenced in evaluation (SeeClick, uGround, ShowUI, etc.) should be downloaded according to their respective repositories.

---

### 🚀 Training

```bash
bash src/open-r1-multimodal/run_scripts/train.sh
```

---

### 📊 Evaluation

To evaluate with the **Qwen2.5-VL-3B** model:

```bash
bash evalue/run_Qwen3B.sh
```

For other model variants, refer to the corresponding scripts in the `evalue/` directory.

---

## 🧪 Ablation Highlights

| Configuration | P-LR Target Avg | E2E-GG-3B Avg |
|:---|:---:|:---:|
| Baseline (Qwen2.5-VL-3B) | 15.31 | 15.31 |
| Random region training | 19.80 | 16.19 |
| Fixed-size Sideal crop | 26.83 | 18.15 |
| Simple prompt (no logical structure) | 24.60 | 23.47 |
| **Full Logical Region (Ours)** | **28.21** | **38.90** |

Two-stage GRPO consistently outperforms both single-stage GRPO and SFT+GRPO hybrids.

---

## 📝 Citation

```bibtex
@article{l2f2026,
  title  = {L2F: Learning to Focus on Logical Regions for High-Resolution GUI Grounding},
  year   = {2026}
}
```

---

<div align="center">

📄 **[Paper (Coming Soon)]** | 🤗 **[Dataset (Coming Soon)]** | 💻 **[Code](https://github.com/yixiunanyangu/L2F-code)**

<sub>Our codes and dataset will be released to promote the development of the community.</sub>

</div>
