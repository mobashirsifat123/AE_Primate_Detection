# AE-Primate: Annotation-Efficient Primate Detection via Lightweight FastKAN and Cost-Aware Active Learning

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](main.pdf)
[![Venue](https://img.shields.io/badge/Venue-IEEE-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code release and reproduction repository for **AE-Primate**, an end-to-end framework coupling fixed-basis Kolmogorov-Arnold Network (KAN) bottlenecks with cost-aware active acquisition for primate monitoring on remote camera-trap networks.

---

## 🌟 Key Highlights

1. **Fixed-Basis FastKAN Architecture (`YOLO11n-F16`)**:
   - Replaces the deepest standard C3k2 convolutional block with a $K=16$ Gaussian Radial Basis Function (RBF) FastKAN bottleneck.
   - **Ranks #1** on held-out camera stations ($0.5626\ m\text{AP}_{50\text{-}95}$, $+1.49\%$ over YOLO11n-B0) at **2.45M parameters** and **8.82 ms FP16 latency** (edge-deployable on NVIDIA Jetson Orin Nano at 32.4 FPS, 15W mode).
2. **Cost-Aware Multi-Modal Active Acquisition Policy**:
   - Synthesizes multi-head detection uncertainty, KAN feature-space core-set diversity, and a crowding-penalised annotation-effort proxy.
   - **Maintains statistical parity in detection accuracy** with passive random sampling ($m\text{AP}_{50\text{-}95} = 0.4618 \pm 0.0019$ on validation, $0.4119 \pm 0.0031$ on held-out test; $\Delta\text{AUBC} = -0.0024$, $p = 0.513$).
3. **Bounding Box Economy & Variance Suppression**:
   - Queries **$63.4$ fewer cumulative bounding boxes** ($1028.2 \pm 7.9$ vs. $1091.6 \pm 26.5$, a **$5.81\%$ reduction**, $p = 0.008$) across 5 active cycles ($n \in [300, 900]$ frames).
   - Reduces box query standard deviation by **$70.2\%$** ($\sigma = 7.9$ vs. $26.5$), eliminating the budget blowout risks caused by sampling dense baboon troops.
4. **Strict Zero-Leakage Location-Disjoint Evaluation**:
   - Zero geographic leakage across 49 stations in Nkhotakota Wildlife Reserve: 35 training stations ($30,620$ frames), 7 validation stations ($5,560$ frames), and 7 held-out test stations ($3,206$ frames).
   - Test stations were strictly sealed during active learning and only unsealed after all 20 trajectories completed.

---

## 📊 Summary of Main Results

### Active Learning Multi-Cycle Benchmark (Protocol v2.0, 5 Seeds: 42, 101, 202, 303, 404)

| Strategy | Cycle 0 ($n=300$) | Cycle 2 ($n=600$) | Cycle 4 ($n=900$) | $\text{AUBC}_{50\text{-}95}$ | Cumulative Boxes | Box Savings vs. Random |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform Random** | $0.3247 \pm 0.0079$ | $0.4214 \pm 0.0098$ | $0.4681 \pm 0.0107$ | $0.4151 \pm 0.0068$ | $1091.6 \pm 26.5$ | Baseline |
| **Uncertainty (Entropy)** | $0.3259 \pm 0.0075$ | $0.4253 \pm 0.0135$ | $0.4665 \pm 0.0089$ | $0.4136 \pm 0.0084$ | $1059.2 \pm 22.8$ | $-32.4$ boxes ($-2.97\%$) |
| **Diversity (CoreSet)** | $0.3394 \pm 0.0097$ | $0.4357 \pm 0.0049$ | $0.4748 \pm 0.0062$ | $0.4226 \pm 0.0050$ | $1080.2 \pm 26.9$ | $-11.4$ boxes ($-1.04\%$) |
| **AE-Primate (Proposed)** | $0.3364 \pm 0.0084$ | $0.4229 \pm 0.0041$ | $0.4618 \pm 0.0019$ | $0.4127 \pm 0.0057$ | **$1028.2 \pm 7.9$** | **$-63.4$ boxes ($-5.81\%$, $p=0.008$)** |

### Sealed Held-Out Test Generalization (3,206 Frames, 7 Unseen Stations)

| Strategy | Test $m\text{AP}_{50\text{-}95}$ | Test $m\text{AP}_{50}$ | Precision | Recall | Generalization Std ($\sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Uniform Random** | $0.4251 \pm 0.0058$ | $0.6867 \pm 0.0076$ | $0.7809$ | $0.6206$ | $0.0058$ |
| **Uncertainty (Entropy)** | $0.4155 \pm 0.0084$ | $0.6799 \pm 0.0104$ | $0.7583$ | $0.6264$ | $0.0084$ |
| **Diversity (CoreSet)** | $0.4300 \pm 0.0045$ | $0.7046 \pm 0.0063$ | $0.8032$ | $0.6359$ | $0.0045$ |
| **AE-Primate (Proposed)** | $0.4119 \pm 0.0031$ | $0.6729 \pm 0.0039$ | $0.7478$ | $0.6261$ | **$0.0031$** |

---

## 📁 Repository Structure

```
paper/
├── main.tex                             # Complete LaTeX source (IEEE format, 10 pages)
├── main.pdf                             # Compiled 10-page camera-ready paper
├── figures/                             # Vector PDF figures & visual overlays
│   ├── figure1_al_efficiency.pdf        # AL accuracy & cumulative box trajectories
│   ├── figure2_pareto_frontier.pdf      # Latency & parameter Pareto frontier
│   ├── figure3_spatial_generalization.pdf # Per-station spatial generalization
│   ├── figure4_qualitative_detections.pdf# Field visual detection overlays
│   └── tables.tex                       # LaTeX tables
├── references.bib                       # Verified BibTeX bibliography
└── README.md                            # Reproduction instructions
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/mobashirsifat123/AE_Primate_Detection.git
cd AE_Primate_Detection
uv sync
```

### 2. Run Test Suite
```bash
uv run pytest tests/
```

### 3. Compile LaTeX Paper to PDF
```bash
bash paper/build_venues.sh
```

---

## 📖 Citation

If you use AE-Primate or our benchmarks in your research, please cite:

```bibtex
@inproceedings{sifat2026aeprimate,
  title={Annotation-Efficient Primate Detection: FastKAN Architectures and the Spatial Generalization Dilemma in Wildlife Active Learning},
  author={Sifat, Mobashir},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
