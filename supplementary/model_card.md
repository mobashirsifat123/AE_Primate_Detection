# Model Card: YOLO11n-F16 Detector

## 1. Model Summary
- **Model Name**: YOLO11n-F16
- **Architecture**: Lightweight single-stage anchor-free detector with fixed-basis Kolmogorov-Arnold Network (FastKAN) spline layers replacing standard convolutions in the deepest bottleneck block (`C3k2` at feature scale $P_5$).
- **Base Detector**: Ultralytics YOLO11n (2.62\,M parameters, 6.3\,GFLOPs).
- **Target Deployment**: Remote, solar-powered camera-trap nodes in miombo forest reserves (e.g., NVIDIA Jetson Orin Nano, Raspberry Pi with Coral Edge TPU).
- **Primary Species Target**: Yellow baboon (*Papio cynocephalus*) and vervet monkey (*Chlorocebus pygerythrus*).

---

## 2. Model Architecture & Specifications

```
                       [Input Frame: 640 x 640 x 3]
                                     │
                    [YOLO11 Backbone: P1 - P4 Conv Blocks]
                                     │
                          [P5 Feature Map: 20 x 20]
                                     │
                   ┌─────────────────┴─────────────────┐
                   │       C3k2_FixedFastKAN Block     │
                   │                                   │
                   │   Linear Residual Base:           │
                   │     y_base = W_base · x           │
                   │                                   │
                   │   Fixed RBF Spline Basis (K=16):  │
                   │     φ_k(x) = exp(-(x - μ_k)² / 2σ²) │
                   │     y_spline = Σ_k c_k · φ_k(x)   │
                   │                                   │
                   │   Fused Output:                   │
                   │     y = y_base + y_spline         │
                   └─────────────────┬─────────────────┘
                                     │
                    [PANet Neck & Decoupled Detection Head]
                                     │
                [Bounding Boxes (CIoU) + Class Probabilities]
```

### Parameter & Computational Complexity
| Specification | Standard YOLO11n (B0) | B-spline KAN (B1) | Adaptive AB-FastKAN | **YOLO11n-F16 (Ours)** | MegaDetector v6 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Parameters** | 2.62\,M | 2.95\,M | 2.52\,M | **2.48\,M** (2,483,792) | 25.53\,M |
| **GFLOPs (640x640)** | 6.3 | 7.1 | 6.4 | **6.4** | 74.2 |
| **Spline Type** | None (CNN) | Cubic B-spline | Hard-Concrete Gate | **Fixed Gaussian RBF** | None (CNN) |
| **Grid Size ($K$)** | N/A | 5 knots | Dynamic | **16 RBF bases** | N/A |
| **Grid Interval** | N/A | $[-1, 1]$ | $[-1, 1]$ | **$[-1, 1]$ (fixed)** | N/A |

---

## 3. Hardware Latency & Profiling

Inference benchmarks were measured directly on server hardware (`profile_adaptive_fastkan.py`):

| Hardware Platform | Operating Envelope | Precision | Mean Latency | Throughput | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA A40** | 300\,W Server PCIe | FP16 TensorRT / Torch | **10.99\,ms** | **91.0\,FPS** | **480\,MB** |
| **NVIDIA A40** | 300\,W Server PCIe | FP32 PyTorch | **14.22\,ms** | **70.3\,FPS** | **620\,MB** |

- **Edge Throughput Readiness**: At 10.99\,ms per frame (91.0\,FPS) on an A40 GPU with sub-500\,MB memory consumption, YOLO11n-F16 is architecturally lightweight for edge inference pipelines. Physical testing on low-power edge nodes (such as NVIDIA Jetson Orin Nano) is identified as future work.

---

## 4. Benchmark Performance & Evaluation

All evaluations are conducted on the **Nkhotakota Primate Dataset** under a zero-leakage, location-disjoint split:
- **Held-Out Test Set** (7 unseen stations, 3,206 frames):
  - **YOLO11n-F16 (Full Dataset)**: **0.5626** $m\text{AP}_{50\text{-}95}$ (0.7684 $m\text{AP}_{50}$)
  - **Standard CNN YOLO11n (B0)**: 0.5477 $m\text{AP}_{50\text{-}95}$ (+1.49% advantage for F16)
  - **B-spline KAN (B1)**: 0.5447 $m\text{AP}_{50\text{-}95}$ (+1.79% advantage for F16)
  - **MegaDetector v6 (Zero-Shot)**: 0.1255 $m\text{AP}_{50\text{-}95}$ (severe domain collapse)

---

## 5. Active Learning Compatibility

YOLO11n-F16 is designed specifically for integration with the **AE-Primate Active Learning Framework**:
- Fixed-basis spline representations provide stable gradient signals in low-data training regimes ($n \in [300, 900]$ images).
- Extracted bottleneck feature representations (256-D) enable high-dimensional core-set diversity sampling without additional embedding heads.
- When trained from pure COCO-80 pretrained weights across 5 active cycles (Cycle 4, $n=900$ frames across $N=5$ seeds):
  - **AE-Primate (Diversity)** achieves top validation AUBC (0.4226) and $0.4300 \pm 0.0045$ test mAP, outperforming passive random ($0.4251 \pm 0.0058$) on 3/5 seeds (paired $t$-test $p = 0.247$).
  - **AE-Primate (Cost-Aware)** achieves a statistically significant reduction of **59.8 bounding boxes ($-5.48\%$, paired $t$-test $p=0.0105$)**, saving 12.0--17.9 minutes of manual labor across 900 frames, and suppressing cross-seed standard deviation by **60.3%** (from $\sigma=26.5$ down to $\sigma=10.5$, 84.2% variance reduction), with a $-1.32$ mAP test accuracy trade-off ($0.4119$ vs. $0.4251$, $p=0.156$).
