import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from ultralytics import YOLO


class DetectorTrainer:
    """
    Standard training driver for the frozen YOLO11n-F16 detector.
    Does not modify Ultralytics internals.
    Guarantees:
    - Points training strictly to the current labeled pool manifest.
    - Evaluates on the frozen location-disjoint validation set.
    """
    def __init__(
        self,
        architecture_yaml: str,
        pretrained_checkpoint: Optional[str] = None,
        device: str = "2",
        workers: int = 4
    ):
        self.architecture_yaml = architecture_yaml
        self.pretrained_checkpoint = pretrained_checkpoint
        self.device = device
        self.workers = workers

    def create_cycle_data_yaml(
        self,
        base_data_yaml_path: str,
        current_train_txt: str,
        output_yaml_path: str,
        val_txt: Optional[str] = None
    ) -> str:
        """
        Creates a cycle-specific data.yaml pointing train to the active learning labeled set,
        while strictly preserving the location-disjoint val/test split.
        """
        with open(base_data_yaml_path, "r") as f:
            base_data = yaml.safe_load(f)

        cycle_data = dict(base_data)
        cycle_data["train"] = str(Path(current_train_txt).resolve())
        if val_txt:
            cycle_data["val"] = str(Path(val_txt).resolve())
            cycle_data["test"] = str(Path(val_txt).resolve())

        out_path = Path(output_yaml_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.safe_dump(cycle_data, f, sort_keys=False)

        return str(out_path)

    def train_cycle(
        self,
        cycle_data_yaml: str,
        output_dir: str,
        run_name: str,
        epochs: int = 30,
        batch: int = 64,
        imgsz: int = 640,
        lr0: float = 0.01,
        optimizer: str = "SGD",
        patience: int = 100,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes YOLO training for one cycle and returns validation results.
        Extracts metrics directly from training results or results.csv,
        avoiding redundant validation passes over network storage.
        """
        if self.pretrained_checkpoint and Path(self.pretrained_checkpoint).exists():
            model = YOLO(self.architecture_yaml).load(self.pretrained_checkpoint)
        else:
            model = YOLO(self.architecture_yaml)

        results = model.train(
            data=cycle_data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            workers=self.workers,
            cache="ram",
            plots=False,
            project=output_dir,
            name=run_name,
            seed=seed,
            deterministic=True,
            optimizer=optimizer,
            lr0=lr0,
            patience=patience,
            exist_ok=True,
            verbose=False
        )

        best_pt = Path(output_dir) / run_name / "weights" / "best.pt"
        last_pt = Path(output_dir) / run_name / "weights" / "last.pt"

        mAP50_95, mAP50, precision, recall = 0.0, 0.0, 0.0, 0.0

        # Method 1: Extract from results object
        if hasattr(results, "box") and results.box is not None:
            try:
                mAP50_95 = float(results.box.map)
                mAP50 = float(results.box.map50)
                precision = float(results.box.mp)
                recall = float(results.box.mr)
            except Exception:
                pass

        # Method 2: Extract from results.csv
        results_csv = Path(output_dir) / run_name / "results.csv"
        if mAP50_95 == 0.0 and results_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(results_csv)
                col_map = {c.strip(): c for c in df.columns}
                last_row = df.iloc[-1]
                if "metrics/mAP50-95(B)" in col_map:
                    mAP50_95 = float(last_row[col_map["metrics/mAP50-95(B)"]])
                if "metrics/mAP50(B)" in col_map:
                    mAP50 = float(last_row[col_map["metrics/mAP50(B)"]])
                if "metrics/precision(B)" in col_map:
                    precision = float(last_row[col_map["metrics/precision(B)"]])
                if "metrics/recall(B)" in col_map:
                    recall = float(last_row[col_map["metrics/recall(B)"]])
            except Exception:
                pass

        # Method 3: Fallback to explicit val if not extracted
        if mAP50_95 == 0.0:
            target_pt = best_pt if best_pt.exists() else last_pt
            val_model = YOLO(str(target_pt))
            val_metrics = val_model.val(
                data=cycle_data_yaml,
                imgsz=imgsz,
                device=self.device,
                split="val",
                workers=self.workers,
                verbose=False
            )
            mAP50_95 = float(val_metrics.box.map)
            mAP50 = float(val_metrics.box.map50)
            precision = float(val_metrics.box.mp)
            recall = float(val_metrics.box.mr)

        return {
            "best_checkpoint": str(best_pt),
            "last_checkpoint": str(last_pt),
            "mAP50_95": mAP50_95,
            "mAP50": mAP50,
            "precision": precision,
            "recall": recall
        }
