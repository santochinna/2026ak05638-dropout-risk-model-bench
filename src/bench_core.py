"""Shared helpers: data loading, model zoo, metric computation."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 5638
LABEL_COL = "Target"
CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]

# display name -> file stem used in model/
MODEL_FILES = {
    "Logistic Regression": "logreg_pipe",
    "Decision Tree": "dtree_pipe",
    "KNN": "knn_pipe",
    "Naive Bayes": "gnb_pipe",
    "Support Vector Machine": "svm_pipe",
}


def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM / tabs / stray spaces from header names."""
    df = df.copy()
    df.columns = [c.replace("﻿", "").replace("\t", "").strip() for c in df.columns]
    return df


def read_student_csv(path: str) -> pd.DataFrame:
    return tidy_columns(pd.read_csv(path, sep=None, engine="python"))


def model_zoo():
    """Estimator + small search grid for each of the five classifiers."""
    def pipe(est):
        return Pipeline([("scale", StandardScaler()), ("clf", est)])
    return {
        "Logistic Regression": (pipe(LogisticRegression(max_iter=5000, random_state=SEED)),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (pipe(DecisionTreeClassifier(random_state=SEED)),
                          {"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [1, 5, 15]}),
        "KNN": (pipe(KNeighborsClassifier()),
                {"clf__n_neighbors": [5, 9, 15, 25], "clf__weights": ["uniform", "distance"]}),
        "Naive Bayes": (pipe(GaussianNB()),
                        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}),
        "Support Vector Machine": (pipe(SVC(probability=True, random_state=SEED)),
                                   {"clf__C": [0.5, 1, 5], "clf__kernel": ["rbf", "linear"]}),
    }


def score_model(pipe, X, y_true, n_classes):
    """Return the six required metrics. y_true holds integer class ids.
    Precision/Recall/F1 are macro-averaged; AUC is one-vs-rest macro."""
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                             labels=list(range(n_classes))),
        "Precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }
