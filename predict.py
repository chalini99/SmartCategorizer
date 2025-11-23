# predict.py
import joblib, json
from pathlib import Path
import numpy as np

MODEL_PATH = Path("artifacts/model_pipeline.joblib")
if not MODEL_PATH.exists():
    raise SystemExit("Model not found. Run: python train_model.py")

model = joblib.load(MODEL_PATH)

# Helper to get top tokens that influenced prediction (using coef_ and tfidf)
def explain(text, top_k=5):
    vec = model.named_steps['tfidf']
    clf = model.named_steps['clf']
    X = vec.transform([text])
    probs = clf.predict_proba(X)[0]
    pred_idx = probs.argmax()
    pred_label = clf.classes_[pred_idx]
    # get feature names and coefficients for the predicted class
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[pred_idx]
    # compute token scores = X * coef (sparse)
    token_indices = X.nonzero()[1]
    token_scores = {feature_names[i]: coefs[i] for i in token_indices}
    # sort tokens by absolute coef impact
    sorted_tokens = sorted(token_scores.items(), key=lambda x: -abs(x[1]))
    top_tokens = [t for t,_ in sorted_tokens[:top_k]]
    return pred_label, probs.max(), top_tokens

def predict_list(texts):
    preds = model.predict(texts)
    probs = model.predict_proba(texts).max(axis=1)
    explanations = [explain(t) for t in texts]
    for t, p, prob in zip(texts, preds, probs):
        label, confidence, top_tokens = explain(t)
        print(f"{t} -> {label} (conf={confidence:.2f}) | tokens: {top_tokens}")

if __name__ == "__main__":
    samples = [
        "Starbucks Coffee 123",
        "AMZ Mktp Prime",
        "Shell Gas Station 45",
        "Walmart Grocery #23",
        "Netflix Subscription POS"
    ]
    predict_list(samples)
