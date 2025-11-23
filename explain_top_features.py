# explain_top_features.py
import joblib
from pathlib import Path
import numpy as np
MODEL_PATH = Path("artifacts/model_pipeline.joblib")
if not MODEL_PATH.exists():
    raise SystemExit("Train model first (python train_model.py)")

model = joblib.load(MODEL_PATH)
vec = model.named_steps['tfidf']
clf = model.named_steps['clf']

feature_names = vec.get_feature_names_out()
classes = clf.classes_
coefs = clf.coef_  # shape (n_classes, n_features)

topk = 10
for i,cls in enumerate(classes):
    top_idx = np.argsort(coefs[i])[-topk:][::-1]
    top_feats = [feature_names[j] for j in top_idx]
    print(f"\nTop features for class [{cls}]:")
    print(top_feats)
