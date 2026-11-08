"""Train all five classifiers, score them on the held-out split, save everything."""
import json, os, sys
import joblib, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
sys.path.insert(0, os.path.dirname(__file__))
import bench_core as bc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = bc.read_student_csv(os.path.join(ROOT, "data", "students_raw.csv"))
feat_cols = [c for c in df.columns if c != bc.LABEL_COL]
y = df[bc.LABEL_COL].map({c: i for i, c in enumerate(bc.CLASS_ORDER)})
X = df[feat_cols]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=bc.SEED)

# held-out file that is uploaded in the Streamlit app (features + original text label)
hold = X_te.copy()
hold[bc.LABEL_COL] = y_te.map(dict(enumerate(bc.CLASS_ORDER)))
hold.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

cv = StratifiedKFold(5, shuffle=True, random_state=bc.SEED)
rows, best_params = [], {}
for name, (pipe, grid) in bc.model_zoo().items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1).fit(X_tr, y_tr)
    est = gs.best_estimator_
    joblib.dump(est, os.path.join(ROOT, "model", bc.MODEL_FILES[name] + ".joblib"))
    m = bc.score_model(est, X_te, y_te, len(bc.CLASS_ORDER))
    rows.append({"ML Model Name": name, **m})
    best_params[name] = gs.best_params_
    print(f"{name:24s}", {k: round(v, 4) for k, v in m.items()}, gs.best_params_)

res = pd.DataFrame(rows).round(4)
res.to_csv(os.path.join(ROOT, "reports", "metrics_summary.csv"), index=False)
json.dump({"features": feat_cols, "classes": bc.CLASS_ORDER, "best_params": best_params},
          open(os.path.join(ROOT, "model", "schema.json"), "w"), indent=2, default=str)
print(res.to_string(index=False))
