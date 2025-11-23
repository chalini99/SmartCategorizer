import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv("data.csv")

# PIE CHART — Category Distribution
plt.figure(figsize=(8,8))
df['category'].value_counts().plot.pie(autopct='%1.1f%%', shadow=True)
plt.title("Category Distribution")
plt.ylabel("")
plt.savefig(ART / "category_distribution.png", dpi=150)
plt.close()

# Load metrics
with open(ART / "metrics.json") as f:
    metrics = json.load(f)

# BAR CHART — Per-class F1
cls_scores = {
    cls: metrics["report"][cls]["f1-score"]
    for cls in metrics["report"]
    if cls not in ("accuracy", "macro avg", "weighted avg")
}

plt.figure(figsize=(10,6))
plt.bar(cls_scores.keys(), cls_scores.values(), color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("Per-Class F1 Scores")
plt.ylabel("F1 Score")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(ART / "f1_scores.png", dpi=150)
plt.close()

print("Analytics charts generated successfully!")
