# AbyssNet-v1: Real-Time Object Detection Based on DEIMv2

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![LICENSE](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![DEIMv2](https://img.shields.io/badge/Based_on-DEIMv2-orange.svg)](https://github.com/Intellindust-AI-Lab/DEIMv2)

AbyssNet-v1 is an enhanced object detector built upon the original [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) repository. It integrates **Attention Memory Residual (AMR)**, **Self-Attention Skip (SA-Skip)**, and **Stochastic Depth (Random Layer Dropout)** to optimize real-time performance. This project leverages DINOv3 STAs (ViT-Tiny) as the backbone, paired with a HybridEncoder and a customized Transformer Decoder to deliver superior detection mAP and faster convergence speeds.


> [!IMPORTANT]
> **Project Status**: This project is currently under review for **NCWIA 2026**.

---

## 🎬 Demo Video

Below is the visualization of the object detection results in action:

https://github.com/user-attachments/assets/5009bcb7-af76-421e-982f-92bdae909f21


*The video `balivideo_Modified.mp4` is hosted on GitHub repository `tch107146/AbyssNet-v1`.*

---

## 📊 Performance Comparison

Experimental results trained on a custom 4-class COCO format dataset for 120 epochs:

| Metric / Model | Baseline DEIMv2 | AbyssNet-v1 (Ours) | Difference / Improvement |
| :--- | :---: | :---: | :---: |
| **Best AP50:95** | 0.7795 (Epoch 114) | **0.7803** (Epoch 114) | **+0.0008** |
| **Best AP50** | 0.9571 (Epoch 114) | **0.9589** (Epoch 101) | **+0.0018** |
| **Best AP75** | 0.8683 | **0.8730** | **+0.0047** |
| **Total training FLOPs** | 11.97 G | **11.01 G** | **-8.33%** |
| **Total training time** | 10h 18m 41s | **8h 56m 44s** | **Fastened by 13.25%** |

> [!NOTE]
> The significant improvement in AP75 (+0.0047) indicates that AMR and SA-Skip help the decoder regression heads localize bounding boxes much more accurately under strict IoU thresholds.

---

## 📁 Repository Structure

```yaml
D:\GIT_TCHv1
├── configs/                   # Configuration files (deimv2_dinov3_s_coco.yml)
├── engine/
│   └── deim/
│       ├── deim.py            # Complete Model Assembler
│       ├── deim_decoder.py    # Modified Decoder with AMR, SA-Skip, and Stochastic Depth
│       └── dfine_decoder.py   # Deformable Attention with Decoupled CA support
├── tools/                     # Utility tools for visualization and inference
├── train.py                   # Training entry point
└── requirements.txt           # Python dependencies
```

---

## 🛠️ Getting Started

### 1. Installation
Ensure you have PyTorch installed (preferably with CUDA support). Install dependencies via pip:
```bash
pip install -r requirements.txt
```

### 2. Configuration
The model hyperparameters can be configured in [configs/deimv2/deimv2_dinov3_s_coco.yml](configs/deimv2/deimv2_dinov3_s_coco.yml). You can enable/disable decoupled CA or Stochastic Depth:
```yaml
DEIMTransformer:
  num_queries: 150
  decoupled: False         # Set True to enable decoupled CA
  drop_path_rate: 0.0      # Set >0.0 to enable Stochastic Depth in training
```

### 3. Training
To start training the AbyssNet-v1 model from scratch or using a checkpoint:
```bash
python train.py -c configs/deimv2/deimv2_dinov3_s_coco.yml --use-amp
```

### 4. Inference & Visualization
You can run inference on your test video or image dataset using the scripts provided in `tools/`:
```bash
python tools/inference/torch_inf_vis.py
```

---

## 📜 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
