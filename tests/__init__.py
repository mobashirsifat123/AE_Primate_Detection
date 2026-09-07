import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "YOLO-KAN").exists():
    sys.path.insert(0, str(ROOT / "YOLO-KAN"))
sys.path.insert(0, str(ROOT))
