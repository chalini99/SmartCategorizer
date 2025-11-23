# retrain_with_feedback.py
import pandas as pd
from pathlib import Path
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# load base data and feedback
base = pd.read_csv("data.csv")
fb = Path("feedback_corrections.csv")
if fb.exists():
    corrections = pd.read_csv(fb)
    # rename to match base columns
    corrections = corrections[['merchant','correct_label']].rename(columns={'correct_label':'category'})
    merged = pd.concat([base, corrections.rename(columns={'merchant':'merchant','category':'category'})], ignore_index=True)
else:
    merged = base

# simple retrain on merged
X = merged['merchant'].astype(str)
y = merged['category'].astype(str)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
    ("clf", LogisticRegression(max_iter=2000))
])
pipeline.fit(X,y)
joblib.dump(pipeline, "artifacts/model_pipeline_retrained.joblib")
print("Retrained model saved to artifacts/model_pipeline_retrained.joblib")
