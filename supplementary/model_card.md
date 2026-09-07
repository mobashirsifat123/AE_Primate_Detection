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
| **Parameters** | 2.62\,M | 2.95\,M | 2.52\,M | **2.45\,M** (-6.5%) | 25.53\,M |
| **GFLOPs (640x640)** | 6.3 | 7.1 | 6.4 | **6.3** | 74.2 |
| **Spline Type** | None (CNN) | Cubic B-spline | Hard-Concrete Gate | **Fixed Gaussian RBF** | None (CNN) |
| **Grid Size ($K$)** | N/A | 5 knots | Dynamic | **16 RBF bases** | N/A |
| **Grid Interval** | N/A | $[-1, 1]$ | $[-1, 1]$ | **$[-1, 1]$ (fixed)** | N/A |

---

## 3. Hardware Latency & Edge Profiling

Inference benchmarks were conducted under realistic field operating envelopes:

| Hardware Platform | Operating Envelope | Precision | Mean Latency | Throughput | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA Jetson Orin Nano** | 15\,W Super Mode | FP16 TensorRT | **30.8\,ms** | **32.4\,FPS** | **442\,MB** |
| **NVIDIA Jetson Orin Nano** | 7\,W Low Power | INT8 TensorRT | **18.4\,ms** | **54.3\,FPS** | **318\,MB** |
| **NVIDIA A40** | 300\,W Server | FP16 TensorRT | **10.99\,ms** | **91.0\,FPS** | **1.12\,GB** |
| **NVIDIA A40** | 300\,W Server | FP32 PyTorch | **14.22\,ms** | **70.3\,FPS** | **1.45\,GB** |

- **Edge Feasibility**: The entire detector footprint easily fits into the 8\,GB unified memory of the Jetson Orin Nano, leaving over 7\,GB for real-time video caching and telemetry services.
- **Energy Budget**: At 32.4\,FPS and 15\,W, per-frame energy consumption is $0.46$\,J, permitting 15,000+ inference events on a compact 20\,Wh solar buffer battery without recharge.

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

YOLO11n-F16 is designed specifically for integration with the **AE-Primate Cost-Aware Active Learning Policy**:
- Fixed-basis spline representations provide stable gradient signals in low-data training regimes ($n \in [300, 900]$ images).
- Extracted bottleneck feature representations enable high-dimensional core-set diversity sampling without additional embedding heads.
- When trained from pure COCO-80 pretrained weights across 5 active cycles, YOLO11n-F16 matches passive random sampling accuracy while saving $63.4$ bounding boxes ($-5.81\%$, $p=0.008$) and cutting box-count variance by $70.2\%$.
