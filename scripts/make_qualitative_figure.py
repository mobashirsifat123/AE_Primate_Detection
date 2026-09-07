#!/usr/bin/env python3
"""
Generate Figure 4: Qualitative detection comparison
Runs inference with F16 (proposed) best checkpoint on selected
challenging frames and saves a comparison grid.
"""
import os, sys, json, random
import numpy as np
from pathlib import Path

# Setup
CHECKPOINT = "/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/experiments/runs/primate_al_pilot/benchmark_proposed/cycle_4/train_f16/weights/best.pt"
VAL_DIR = "/home/moba/data/primate_yolo/images/val"
OUTPUT = "/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/paper_outputs/figure4_qualitative_detections.pdf"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Try to import YOLO
sys.path.insert(0, "/mnt/nas/users/moba/projects/AE_Primate_Detection/worktrees/primate-al-pilot/YOLO-KAN")
try:
    from ultralytics import YOLO
    model = YOLO(CHECKPOINT)
    MODEL_LOADED = True
    print("[OK] YOLO model loaded")
except Exception as e:
    print(f"[WARN] Could not load YOLO: {e}")
    MODEL_LOADED = False

# Select diverse frames: pick frames with annotations
import glob
val_images = sorted(glob.glob(f"{VAL_DIR}/*.jpg"))
random.seed(42)
selected = random.sample(val_images, min(6, len(val_images)))
print(f"Selected {len(selected)} frames for visualization")

def draw_boxes(ax, img_path, boxes_xyxy, labels, colors, title="", confs=None):
    """Draw image with bounding boxes."""
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
    except Exception:
        ax.imshow(np.zeros((640, 640, 3), dtype=np.uint8) + 128)
    ax.set_title(title, fontsize=8, pad=2)
    ax.axis("off")
    h, w = 640, 640
    try:
        from PIL import Image
        img_arr = Image.open(img_path)
        w, h = img_arr.size
    except: pass
    for i, (box, label) in enumerate(zip(boxes_xyxy, labels)):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=1.5, edgecolor=colors[i % len(colors)],
                                   facecolor='none')
        ax.add_patch(rect)
        conf_str = f" {confs[i]:.2f}" if confs else ""
        ax.text(x1, y1-2, f"{label}{conf_str}", color='white', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i % len(colors)], alpha=0.7))

CLASS_COLORS = {"baboon": "#2ca02c", "vervet": "#1f77b4", "unknown": "#d62728"}
CLASS_NAMES = ["baboon", "vervet"]

fig, axes = plt.subplots(len(selected), 2, figsize=(8, 3 * len(selected)))
if len(selected) == 1:
    axes = [axes]

plt.rcParams.update({"font.size": 8})

for row_idx, img_path in enumerate(selected):
    fname = os.path.basename(img_path)
    ax_img  = axes[row_idx][0]
    ax_pred = axes[row_idx][1]

    # Raw image
    try:
        from PIL import Image
        img = Image.open(img_path)
        ax_img.imshow(img)
        ax_img.set_title(f"Input: {fname}", fontsize=7, pad=2)
        ax_img.axis("off")
    except Exception as e:
        ax_img.set_title(f"No image: {e}", fontsize=6)
        ax_img.axis("off")

    # F16 predictions
    if MODEL_LOADED:
        try:
            results = model(img_path, conf=0.25, verbose=False)
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else []
            clss  = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) > 0 else []
            confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else []
            labels = [CLASS_NAMES[c] if c < len(CLASS_NAMES) else "obj" for c in clss]
            colors = [CLASS_COLORS.get(l, "#d62728") for l in labels]
            draw_boxes(ax_pred, img_path, boxes, labels, colors,
                       title=f"F16 ({len(boxes)} dets, conf≥0.25)", confs=confs)
        except Exception as e:
            ax_pred.set_title(f"Inference error: {e}", fontsize=6)
            ax_pred.axis("off")
    else:
        ax_pred.set_title("Model not available", fontsize=7)
        ax_pred.axis("off")

plt.suptitle("Qualitative Detections: YOLO11n-F16 (PrimateScope)\nNkhotakota Held-Out Validation Stations",
             fontsize=10, y=1.01)
plt.tight_layout()

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight", dpi=200)
fig.savefig(OUTPUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
print(f"[OK] Saved qualitative figure to {OUTPUT}")
