# AE-Primate: Supplementary Material & Replication Guide

This archive contains the official supplementary material, model cards, hyperparameter ledgers, and reproduction scripts for **AE-Primate**: *Annotation-Efficient Primate Detection via Lightweight FastKAN and Cost-Aware Active Learning*.

---

## 1. Overview & Directory Structure

```text
supplementary/
├── README.md               # This reproduction and environment guide
├── model_card.md           # YOLO11n-F16 architecture, latency, and FLOP analysis
├── hyperparameters.json   # Full training and active acquisition parameters
└── checksums.sha256        # Cryptographic SHA-256 ledger for splits and results
```

All experimental results adhere strictly to **Protocol v2.0**:
- **Location-Disjoint Splits**: 49 camera stations across Nkhotakota Wildlife Reserve partitioned into 35 training stations (31,977 frames), 7 validation stations (5,560 frames), and 7 sealed test stations (3,206 frames) with zero geographical overlap.
- **Decontaminated Initialization**: At every active cycle $t \in \{0, 1, 2, 3, 4\}$, models are trained from pure COCO-80 pretrained weights (`yolo11n.pt`) with zero pre-exposure to wildlife annotations, preventing optimization history leakage.
- **Replicated Multi-Seed Benchmark**: All 4 acquisition strategies (Random, Uncertainty, Diversity, AE-Primate) are replicated across 5 independent random seeds ($N=5$: 42, 101, 202, 303, 404), generating 20 independent trajectories and 100 trained model cycles.

---

## 2. Hardware and Environment Requirements

### Compute Environments
- **Server Training**: NVIDIA A40 (48GB VRAM, PCIe Gen4, CUDA 12.4, Driver 535+).
- **Edge Deployment**: NVIDIA Jetson Orin Nano (8GB 128-bit LPDDR5, 15W mode, JetPack 6.0, TensorRT 8.6).

### Software Dependencies
- Python 3.10+
- PyTorch >= 2.5.0, Torchvision >= 0.20.0
- Ultralytics YOLOv11 (v8.3+)
- NumPy, SciPy, Matplotlib, PyYAML

### Quick Environment Setup
```bash
# Clone the repository (anonymized review mirror)
git clone https://anonymous.4open.science/r/AE_Primate_Detection-anon.git
cd AE_Primate_Detection-anon

# Option A: Using uv (recommended)
uv sync

# Option B: Using standard pip virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Step-by-Step Reproduction

### Step 1: Verify Checksums and Splits
Verify that the dataset partitions and benchmark evaluation manifests match the published cryptographic ledger:
```bash
sha256sum -c paper/supplementary/checksums.sha256
```

### Step 2: Run Protocol v2.0 Active Learning
To execute an active learning trajectory for a specific strategy and seed:
```bash
# Example: Proposed AE-Primate policy on seed 42
python -m src.experiments.run_protocol_v2 \
    --strategy proposed \
    --seed 42 \
    --device cuda:1 \
    --cycles 5 \
    --budget-init 300 \
    --budget-step 150
```

To run all 20 trajectories across the 4 strategies and 5 random seeds:
```bash
bash launch_benchmarks.sh
```

### Step 3: Evaluate Sealed Held-Out Test Set
Evaluate the final Cycle 4 models on the sealed 7 held-out camera stations (3,206 frames):
```bash
python -m src.experiments.evaluate_sealed_test \
    --results-dir experiments/results_v2 \
    --output-dir experiments/sealed_test_eval_v2
```

### Step 4: Regenerate Publication Figures
To render vector PDFs (300 DPI) and high-resolution PNGs for Figures 1 and 2:
```bash
python tools/generate_publication_figures_v2.py
```
Outputs:
- `paper/figures/figure1_al_efficiency.pdf` / `.png`
- `paper/figures/figure2_pareto_frontier.pdf` / `.png`

### Step 5: Compile Manuscripts
To build both the camera-ready and double-blind manuscripts:
```bash
bash paper/build_venues.sh
```
Outputs:
- `paper/main.pdf` (Camera-ready, strictly 10 pages)
- `paper/main_anonymous.pdf` (Double-blind review copy, strictly 10 pages)

---

## 4. Statistical Verification Tests

Run the automated test suite to verify statistical claims, box count differences, and test results:
```bash
uv run pytest tests/
```
All 92+ tests verify:
1. Zero geographic overlap between training and evaluation stations.
2. Exact matching of reported AUBC and mAP numbers to raw JSON logs.
3. Statistically significant box savings ($-63.4$ boxes, $-5.81\%$, paired $t(4)=4.88$, $p=0.008$).
4. 70.2% variance suppression ($\sigma=7.9$ vs. $26.5$).
