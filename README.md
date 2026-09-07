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
AE_Primate_Detection/
├── paper/                                   # 10-page camera-ready & anonymous review manuscripts
│   ├── main.tex                             # Complete LaTeX source (IEEEtran format, 10 pages)
│   ├── main.pdf                             # Compiled 10-page camera-ready paper
│   ├── main_anonymous.pdf                   # Double-blind review version
│   ├── figures/                             # Vector publication figures (PDF & PNG)
│   ├── references.bib                       # Verified BibTeX bibliography (37 citations)
│   └── build_venues.sh                      # Zero-warning dual-venue compilation script
├── models/                                  # FastKAN architecture definitions & checkpoints
│   ├── yamls/                               # Architecture YAMLs (yolo11n-F16, F8, F4, C0)
│   ├── modules/                             # FastKAN bottlenecks (Fixed & Adaptive RBF splines)
│   └── checkpoints/                         # Official pretrained & cycle-4 weights (.pt)
│       ├── ae_primate_yolo11n_f16_final.pt  # Final Cycle-4 model checkpoint (4.9 MB)
│       └── yolo11n.pt                       # Standard COCO initialization weights (5.4 MB)
├── primate_al/                              # Core Active Learning Library
│   ├── methods/                             # Samplers: Random, Uncertainty, CoreSet, Proposed Cost-Aware
│   ├── pool.py                              # Active learning candidate & labeled pool manager
│   ├── metadata.py                          # Camera-trap burst event & station metadata manager
│   ├── feature_extractor.py                 # FastKAN ROI feature representation extractor
│   ├── uncertainty.py                       # Multi-head detection predictive entropy scorer
│   └── trainer.py                           # YOLO training & evaluation harness
├── configs/                                 # Active learning & detector configs
│   ├── active_learning/
│   │   ├── protocol_v2_reproduce.yaml       # Portable relative path configuration
│   │   └── protocol_v2.yaml                 # Multi-GPU cluster training configuration
├── experiments/                             # Complete verified empirical records
│   ├── results_v2/                          # All 20 trajectories across 5 seeds (42, 101, 202, 303, 404)
│   │   ├── protocol_v2_summary.json         # Aggregate multi-seed benchmark metrics
│   │   └── protocol_v2_box_counts.json      # Bounding box economy records
│   └── sealed_test_eval_v2/                 # Sealed held-out test evaluations (3,206 frames)
├── scripts/                                 # Active learning dispatch & execution scripts
│   ├── run_active_learning_cycle.py         # Main active learning execution script
│   ├── compute_multiseed_statistics.py      # Multi-seed AUBC & box variance calculator
│   └── launch_protocol_v2_all_gpus.sh       # 4-GPU parallel execution dispatcher
├── tools/                                   # Analysis, evaluation & figure generation
│   ├── evaluate_sealed_test.py              # Zero-leakage held-out test evaluator
│   ├── compile_clean_benchmark.py           # Benchmark tables generator
│   └── generate_publication_figures_v2.py   # Regenerate publication vector figures
├── tests/                                   # Comprehensive unit & integration tests
│   ├── test_active_learning.py              # Tests for AL pool, burst grouping, samplers
│   └── test_adaptive_fastkan.py             # Tests for FastKAN modules & architecture YAMLs
├── supplementary/                           # Model cards, hyperparameters & checksums
│   ├── model_card.md                        # Formal model card documentation
│   ├── hyperparameters.json                 # Comprehensive hyperparameter configuration
│   └── checksums.sha256                     # SHA-256 integrity hashes (39/39 verified)
├── export_adaptive_fastkan.py               # FastKAN continuous-to-fixed compaction utility
├── LICENSE                                  # MIT License
└── README.md                                # Reproduction guide & documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

```bash
git clone https://github.com/mobashirsifat123/AE_Primate_Detection.git
cd AE_Primate_Detection

# Install dependencies (PyTorch, Ultralytics, NumPy, Matplotlib)
pip install torch torchvision torchaudio
pip install -e YOLO-KAN/
pip install scipy matplotlib pandas
```

### 2. Verify System & Run Unit Tests

Execute the full 13-test test suite covering Active Learning samplers, candidate pools, burst redundancy penalties, and FastKAN architecture parsers:

```bash
python3 -m unittest discover -s tests
```

### 3. Verify SHA-256 Checksums

Ensure all empirical logs, figure artifacts, and model checkpoints match the verified paper records:

```bash
shasum -c supplementary/checksums.sha256
```

### 4. Regenerate Manuscript Figures & Empirical Tables

```bash
# Regenerate Table 1, Table 2 and summary statistics
python3 tools/compile_clean_benchmark.py

# Regenerate Figures 1, 2, 3, and 4 (PDF & PNG)
python3 tools/generate_publication_figures_v2.py
```

### 5. Run Active Learning Experiments

To run a single active learning trajectory with the proposed cost-aware policy:

```bash
python3 scripts/run_active_learning_cycle.py \
    --config configs/active_learning/protocol_v2_reproduce.yaml \
    --strategy proposed \
    --total-cycles 4 \
    --initial-budget 300 \
    --cycle-budget 150 \
    --seed 42 \
    --run-dir experiments/runs/reproduce_proposed_seed42
```

### 6. Compile the Camera-Ready Manuscript to PDF

```bash
bash paper/build_venues.sh
```
Outputs `paper/main.pdf` (camera-ready) and `paper/main_anonymous.pdf` (anonymous double-blind).

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
