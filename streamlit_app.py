"""Streamlit front-end: evaluate pre-trained classifiers on an uploaded test CSV (no retraining)."""
import json, os, sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import bench_core as bc

st.set_page_config(page_title="Dropout Risk Model Bench", page_icon="🎓", layout="wide")


@st.cache_resource
def load_artifacts():
    schema = json.load(open(os.path.join(HERE, "model", "schema.json")))
    models = {n: joblib.load(os.path.join(HERE, "model", f + ".joblib")) for n, f in bc.MODEL_FILES.items()}
    return schema, models


schema, models = load_artifacts()
classes, feats = schema["classes"], schema["features"]

st.title("🎓 Student Dropout Risk - Model Bench")
st.caption("Pre-trained classifiers (UCI: Predict Students' Dropout and Academic Success). "
           "Upload a test CSV containing the feature columns and the `Target` column. Models are not retrained.")

with st.sidebar:
    st.header("Controls")
    up = st.file_uploader("Upload test CSV", type=["csv"])
    choice = st.selectbox("Choose a model", list(models.keys()))
    st.markdown("---")
    st.write("Classes:", ", ".join(classes))
    st.write(f"Features expected: {len(feats)}")

if up is None:
    st.info("Upload a test CSV from the sidebar to begin. Use `test_data.csv` from the GitHub repo.")
    st.stop()

try:
    data = bc.tidy_columns(pd.read_csv(up, sep=None, engine="python"))
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

missing = [c for c in feats + [bc.LABEL_COL] if c not in data.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

bad = sorted(set(data[bc.LABEL_COL].astype(str)) - set(classes))
if bad:
    st.error(f"Unexpected values in `{bc.LABEL_COL}`: {bad}. Allowed: {classes}")
    st.stop()

X = data[feats]
y = data[bc.LABEL_COL].map({c: i for i, c in enumerate(classes)})
st.success(f"Loaded {len(data):,} rows.")
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20), use_container_width=True)


def metrics_for(name):
    return bc.score_model(models[name], X, y, len(classes))


# ---- selected model
st.subheader(f"Results: {choice}")
m = metrics_for(choice)
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
pred = models[choice].predict(X)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=classes, ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    rep = classification_report(y, pred, target_names=classes, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# ---- all models
st.subheader("All five models on this upload")
table = pd.DataFrame({n: metrics_for(n) for n in models}).T.round(4)
table.index.name = "ML Model Name"
st.dataframe(table.style.highlight_max(axis=0, color="#cfe8cf"), use_container_width=True)
st.bar_chart(table[["Accuracy", "F1", "MCC"]])
