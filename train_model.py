import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import shutil

OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

# Load config
with open("config.json") as f:
    cfg = json.load(f)
CATEGORIES = cfg["categories"]

# Load dataset
df = pd.read_csv("data.csv")
df["merchant"] = df["merchant"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip()

df = df[df["category"].isin(CATEGORIES)].reset_index(drop=True)

STATIC_ART = Path("static/artifacts")
STATIC_ART.mkdir(parents=True, exist_ok=True)

shutil.copy(OUT_DIR / "category_distribution.png", STATIC_ART / "category_distribution.png")
shutil.copy(OUT_DIR / "f1_scores.png", STATIC_ART / "f1_scores.png")



# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["merchant"], df["category"], test_size=0.18,
    random_state=42, stratify=df["category"]
)

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=2000))
])

print("Training model…")
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, OUT_DIR / "model_pipeline.joblib")
print("Model saved to artifacts/model_pipeline.joblib")

# Predictions
y_pred = pipeline.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average="macro")

# Save classification report
report_text = classification_report(y_test, y_pred)
with open(OUT_DIR / "classification_report.txt", "w") as f:
    f.write(report_text)

# Save metrics.json (IMPORTANT)
report_dict = classification_report(y_test, y_pred, output_dict=True)

metrics = {
    "macro_f1": float(macro_f1),
    "report": report_dict
}

with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
print("HEATMAP DEBUG:", heat_html)

plt.figure(figsize=(9,7))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45, ha="right")
plt.yticks(range(len(CATEGORIES)), CATEGORIES)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(int(cm[i,j])), ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

print("\nMacro F1 Score:", macro_f1)
print("Training complete!")
