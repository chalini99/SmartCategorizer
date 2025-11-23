# app.py – Smart CategorizerAI (FINAL MERGED VERSION)
import os
import uuid
import io
import csv
import json
import random
import html
import subprocess
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for headless systems (prevents GUI errors)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (
    Flask, render_template, request, redirect, session,
    jsonify, send_file, url_for
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Flask setup
# -------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey123"

# -------------------------
# Paths
# -------------------------
MODEL_PATH = Path("artifacts/model_pipeline.joblib")
CONFIG_PATH = Path("config.json")
DATA_PATH = Path("data.csv")
ARTIFACTS_DIR = Path("artifacts")
FEEDBACK_DIR = Path("feedback")
HEATMAP_DIR = Path("static/heatmaps")

ARTIFACTS_DIR.mkdir(exist_ok=True)
FEEDBACK_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load model
# -------------------------
model = None
try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print("[INFO] Model loaded.")
    else:
        print("[WARN] No model found at", MODEL_PATH)
except Exception as e:
    print("[ERROR] Failed to load model:", e)
    model = None

# -------------------------
# Utility helpers
# -------------------------
def confidence_color(conf):
    """Return simple label color name for a confidence score (0..1)."""
    try:
        conf = float(conf)
    except Exception:
        return "red"
    if conf >= 0.8:
        return "green"
    if conf >= 0.5:
        return "orange"
    return "red"

# -------------------------
# explain() - core inference
# -------------------------
def explain(text, top_k=5):
    """
    Return (pred_label, confidence, top_tokens)
    """
    if model is None:
        return "ModelMissing", 0.0, []

    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    X = vec.transform([text])

    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = str(clf.classes_[pred_idx])

    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[pred_idx]

    token_indices = X.nonzero()[1]
    token_scores = {feature_names[i]: float(coefs[i]) for i in token_indices}

    sorted_tokens = sorted(token_scores.items(), key=lambda x: -abs(x[1]))
    top_tokens = [t for t, _ in sorted_tokens[:top_k]]

    return pred_label, float(probs.max()), top_tokens

# -------------------------
# IMAGE HEATMAP GENERATOR
# -------------------------
def generate_heatmap_image(text):
    """
    Creates PNG heatmap for token importance and saves it to static/heatmaps/.
    Returns filename or None if nothing to visualize.
    """
    if model is None:
        return None

    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    X = vec.transform([text])
    feature_names = vec.get_feature_names_out()

    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    coefs = clf.coef_[pred_idx]

    token_indices = X.nonzero()[1]

    token_scores = {}
    for idx in token_indices:
        fname = feature_names[idx]
        token_scores[fname] = float(coefs[idx] * X[0, idx])

    if len(token_scores) == 0:
        return None

    # Sort tokens by contribution magnitude
    items = sorted(token_scores.items(), key=lambda x: -abs(x[1]))
    tokens, scores = zip(*items)

    colors = ["green" if s >= 0 else "red" for s in scores]

    plt.figure(figsize=(6, max(3, len(tokens)*0.35)))
    y = np.arange(len(tokens))

    plt.barh(y, scores, color=colors)
    plt.yticks(y, tokens, fontsize=9)
    plt.title("Token Contribution Heatmap", fontsize=11)
    plt.xlabel("Importance Score", fontsize=9)
    plt.tight_layout()

    filename = f"{uuid.uuid4().hex}.png"
    out_path = HEATMAP_DIR / filename
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("[DEBUG] Saved heatmap to", out_path)
    return filename

# -------------------------
# Semantic index (for suggestions)
# -------------------------
_tf_vectorizer = None
_tf_index = None
_index_corpus = []

def build_tf_index():
    global _tf_vectorizer, _tf_index, _index_corpus
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            _index_corpus = df["merchant"].astype(str).tolist()
        except Exception:
            _index_corpus = []
    else:
        _index_corpus = []

    if len(_index_corpus) == 0:
        _tf_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        _tf_index = _tf_vectorizer.fit_transform(["dummy"])
    else:
        _tf_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        _tf_index = _tf_vectorizer.fit_transform(_index_corpus)

# Build at startup
try:
    build_tf_index()
    print("[INFO] Semantic index built with", len(_index_corpus), "merchants.")
except Exception as e:
    print("[WARN] Failed building semantic index:", e)

# -------------------------
# ROUTES
# -------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    tokens = None

    if request.method == "POST":
        text = request.form.get("merchant", "").strip()
        if text:
            prediction, confidence, tokens = explain(text)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           confidence_color=confidence_color,
                           tokens=tokens)

# Login / logout / admin
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("username") == "admin" and request.form.get("password") == "1234":
            session["admin"] = True
            return redirect("/admin")
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect("/")

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if not session.get("admin"):
        return redirect("/login")

    cfg = {"categories": []}
    if CONFIG_PATH.exists():
        try:
            cfg = json.load(open(CONFIG_PATH))
        except Exception:
            cfg = {"categories": []}

    saved = False
    if request.method == "POST":
        new_list = [c.strip() for c in request.form.get("categories","").split(",") if c.strip()]
        cfg["categories"] = new_list
        json.dump(cfg, open(CONFIG_PATH, "w"), indent=4)
        saved = True

    return render_template("admin.html", category_text=", ".join(cfg.get("categories", [])), saved=saved)

# Analytics
@app.route("/analytics")
def analytics():
    try:
        metrics = json.load(open(ARTIFACTS_DIR / "metrics.json"))
        macro_f1 = metrics.get("macro_f1")
    except Exception:
        macro_f1 = None
    return render_template("analytics.html", macro_f1=macro_f1)

# -------------------------
# Batch upload with interactive summary (Option A)
# -------------------------
@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "POST":
        if "csv_file" not in request.files:
            return "No file uploaded", 400
        file = request.files["csv_file"]
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return f"Failed to read CSV: {e}", 400

        if "merchant" not in df.columns:
            return "CSV must contain 'merchant' column", 400

        preds = []
        for m in df["merchant"].astype(str).tolist():
            label, conf, _ = explain(m)
            preds.append([m, label, round(conf, 4)])

        out_df = pd.DataFrame(preds, columns=["merchant", "prediction", "confidence"])

        # Save latest predictions for insights
        out_df.to_csv(ARTIFACTS_DIR / "predictions_latest.csv", index=False)

        # Prepare quick summary
        summary = {
            "total_rows": len(out_df),
            "by_category": out_df["prediction"].value_counts().to_dict(),
            "avg_confidence": float(out_df["confidence"].mean())
        }

        preview_html = out_df.head(20).to_html(classes="table table-sm", index=False, escape=False)

        return render_template("batch_summary.html", summary=summary, preview=preview_html)

    return render_template("batch.html")

@app.route("/download_predictions")
def download_predictions():
    p = ARTIFACTS_DIR / "predictions_latest.csv"
    if not p.exists():
        return "No predictions available", 404
    return send_file(str(p), as_attachment=True, download_name="predictions_latest.csv")

# Feedback
@app.route("/feedback", methods=["POST"])
def feedback():
    merchant = request.form.get("merchant")
    correct = request.form.get("correct")
    if not merchant or correct is None:
        return "Bad Request", 400
    fb_file = FEEDBACK_DIR / "feedback.csv"
    exists = fb_file.exists()
    with open(fb_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["merchant", "correct_label"])
        writer.writerow([merchant, correct])
    return "OK"

# Retrain
@app.route("/retrain")
def retrain():
    try:
        subprocess.run(["python", "train_model.py"], check=True)
        global model
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            # rebuild semantic index after retrain (if data changed)
            try:
                build_tf_index()
            except Exception:
                pass
        return "Model retrained!", 200
    except subprocess.CalledProcessError as e:
        return f"Retrain failed: {e}", 500

# API Predict
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not request.is_json:
        return jsonify({"error":"Expected JSON with 'merchant' field"}), 400
    text = request.get_json().get("merchant","")
    label, conf, tokens = explain(text)
    return jsonify({"prediction": label, "confidence": conf, "tokens": tokens})

# Suggest API for autocompletion (uses semantic index)
@app.route("/api/suggest", methods=["POST"])
def api_suggest():
    if not request.is_json:
        return jsonify({"results": []})
    q = request.get_json().get("q", "").strip()
    if not q or _tf_vectorizer is None:
        return jsonify({"results": []})
    try:
        vec = _tf_vectorizer.transform([q])
        sims = cosine_similarity(vec, _tf_index).flatten()
        top_idx = sims.argsort()[::-1][:8]
        results = [{"merchant": _index_corpus[i], "score": float(sims[i])} for i in top_idx]
        return jsonify({"results": results})
    except Exception as e:
        print("[WARN] suggest failed:", e)
        return jsonify({"results": []})

# Semantic search page
@app.route("/semantic", methods=["GET", "POST"])
def semantic():
    query = None
    results = []
    if request.method == "POST":
        query = request.form.get("merchant", "").strip()
        if query and _tf_vectorizer is not None:
            vec = _tf_vectorizer.transform([query])
            sims = cosine_similarity(vec, _tf_index).flatten()
            top_idx = sims.argsort()[::-1][:15]
            for i in top_idx:
                m = _index_corpus[i]
                label, conf, _ = explain(m)
                results.append({"merchant": m, "score": round(float(sims[i]),3), "label": label, "conf": round(conf,3)})
    return render_template("semantic.html", query=query, results=results)

# Insights dashboard
@app.route("/insights")
def insights():
    preds_path = ARTIFACTS_DIR / "predictions_latest.csv"
    if preds_path.exists():
        df = pd.read_csv(preds_path)
    else:
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
        else:
            df = pd.DataFrame(columns=["merchant","amount","category"])

    # Ensure numeric amount column exists
    if "amount" not in df.columns:
        df["amount"] = 0.0
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    cat_totals = {}
    try:
        cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False).to_dict()
    except Exception:
        cat_totals = {}

    cat_counts = df["category"].value_counts().to_dict() if "category" in df.columns else {}
    top_merchants = df["merchant"].value_counts().head(10).to_dict()
    total_spend = round(float(df["amount"].sum()), 2)

    return render_template("insights.html",
                           cat_totals=cat_totals,
                           cat_counts=cat_counts,
                           top_merchants=top_merchants,
                           total_spend=total_spend)

# Explain heat (image version)
@app.route("/explain_heat", methods=["GET", "POST"])
def explain_heat():
    pred = None
    conf = None
    heat_image = None
    if request.method == "POST":
        text = request.form.get("merchant","").strip()
        if text:
            pred, conf, _ = explain(text)
            heat_image = generate_heatmap_image(text)
    return render_template("explain_heat.html", pred=pred, conf=conf, heat_image=heat_image)

# Tester (noise robustness)
def generate_noisy_variants(text, n=8):
    variants = set()
    for _ in range(n*3):
        s = text
        if random.random() < 0.4:
            s = s.upper() if random.random() < 0.5 else s.lower()
        if random.random() < 0.3:
            s = s.replace("a", "@").replace("o", "0")
        if random.random() < 0.25:
            s = s.replace(" ", "  ")
        if random.random() < 0.15:
            s = s + " POS"
        if random.random() < 0.10 and len(s) > 1:
            i = random.randint(0, max(0, len(s)-1))
            s = s[:i] + random.choice("!@#") + s[i+1:]
        variants.add(s)
        if len(variants) >= n:
            break
    return list(variants)

@app.route("/tester", methods=["GET","POST"])
def tester():
    results = None
    stability = None
    if request.method == "POST":
        text = request.form.get("merchant","").strip()
        variants = generate_noisy_variants(text, n=10)
        preds = []
        for v in [text] + variants:
            label, conf, tokens = explain(v)
            preds.append({"text": v, "label": label, "conf": round(conf,3), "tokens": tokens})
        orig_label = preds[0]["label"] if preds else None
        same = sum(1 for p in preds if p["label"] == orig_label)
        stability = round(same / len(preds), 3) if preds else None
        results = preds
    return render_template("tester.html", results=results, stability=stability)

# Main
if __name__ == "__main__":
    app.run(debug=True)
