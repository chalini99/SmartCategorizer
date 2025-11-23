# evaluate.py
import json
from pathlib import Path
from pprint import pprint

OUT_DIR = Path("artifacts")
metrics_path = OUT_DIR / "metrics.json"
report_path = OUT_DIR / "classification_report.txt"
cm_path = OUT_DIR / "confusion_matrix.png"

if not metrics_path.exists():
    print("No metrics found. Run train_model.py first.")
    raise SystemExit

with open(metrics_path) as f:
    metrics = json.load(f)

print("=== EVALUATION SUMMARY ===")
print("Macro F1:", metrics.get("macro_f1"))
print("\nPer-class metrics (excerpt):")
pprint({k: metrics['report'][k] for k in metrics['report'] if k not in ("accuracy","macro avg","weighted avg")})
print("\nArtifacts:")
print(" -", report_path)
print(" -", cm_path)

