# Dropout Risk Model Bench

Five-classifier benchmark on the UCI *Predict Students' Dropout and Academic Success* dataset, with a Streamlit app that evaluates the saved models on an uploaded test CSV.

## a. Problem statement
Predict, from information known at enrolment and after the first two semesters, whether a higher-education student will **Dropout**, remain **Enrolled**, or **Graduate** (3-class classification). Early identification of at-risk students lets institutions intervene sooner. The goal is to train and compare five classical classifiers on the same data and expose them through an interactive web app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository, dataset 697 - https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **Size:** 4,424 instances, 36 input features, 1 target (`Target`); no missing values.
- **Features:** demographic (age at enrollment, gender, marital status, nationality), socio-economic (parents' qualification/occupation, scholarship, debtor, tuition status), academic path (application mode, course, previous qualification and grade, admission grade) and academic performance (curricular units credited / enrolled / evaluated / approved / grade for 1st and 2nd semester), plus macro-economic indicators (unemployment, inflation, GDP).
- **Classes:** Graduate 2,209 | Dropout 1,421 | Enrolled 794 (imbalanced, so macro averaging is used).
- **Split:** stratified 80/20 (3,539 train / 885 test, seed 5638). The 885 test rows are in `test_data.csv`.
- **Preprocessing (inside every saved pipeline):** header clean-up; one-hot encoding of the 9 code-like columns (marital status, application mode, course, previous qualification, nationality, parents' qualification and occupation); `StandardScaler` on the remaining numeric columns; target label-encoded as Dropout=0, Enrolled=1, Graduate=2. Naive Bayes additionally uses PCA to decorrelate features.
- **Tuning:** 5-fold stratified `GridSearchCV` on the training set only, scored by macro-F1; the grid includes `class_weight` (none vs balanced) for Logistic Regression, Decision Tree and SVM.
- **Metric averaging:** Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro; MCC is the multi-class version.

## c. GitHub repository link
https://github.com/santochinna/2026ak05638-dropout-risk-model-bench

Live app: https://2026ak05638-ml-assignment-2.streamlit.app/

```
├── streamlit_app.py        # Streamlit front-end
├── requirements.txt
├── README.md
├── test_data.csv           # held-out test set (885 rows) to upload in the app
├── data/students_raw.csv   # full original dataset
├── model/                  # saved pipelines (*.joblib) + schema.json
├── src/bench_core.py       # data loading, model zoo, metric function
├── src/run_training.py     # trains, tunes, evaluates and saves all models
└── reports/metrics_summary.csv
```
Run locally: `pip install -r requirements.txt`, then `streamlit run streamlit_app.py`. To retrain: `python src/run_training.py`.

## d. Models used and results (on the 885-row test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7503 | 0.8774 | 0.7109 | 0.7146 | 0.7094 | 0.6016 |
| Decision Tree | 0.7356 | 0.8398 | 0.6911 | 0.6690 | 0.6762 | 0.5664 |
| KNN | 0.7062 | 0.8110 | 0.6482 | 0.6051 | 0.6132 | 0.5073 |
| Naive Bayes | 0.6870 | 0.7949 | 0.6027 | 0.5825 | 0.5831 | 0.4739 |
| Support Vector Machine | 0.7480 | 0.8797 | 0.7089 | 0.7148 | 0.7073 | 0.6023 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best accuracy (0.750), precision (0.711) and macro-F1 (0.709) by a hair. One-hot encoding of code-like columns plus balanced class weights lifted Enrolled recall from 0.26 to 0.59 while keeping Graduate (0.82) and Dropout (0.74) strong. Simple, fast and nearly tied with SVM. |
| Decision Tree | Depth 4 was the best setting, giving an interpretable model (accuracy 0.736, F1 0.676). Handles the Enrolled class better than KNN/NB (recall 0.45) but its hard splits give coarse probabilities, so AUC (0.840) trails the linear models. |
| KNN | Weak with 36 mixed features: after one-hot encoding the space is high-dimensional and distances lose meaning (accuracy 0.706, AUC 0.811). Enrolled recall is only 0.25 because neighbours are dominated by the larger classes. |
| Naive Bayes | Weakest model (accuracy 0.687, MCC 0.474). Features such as 1st/2nd-semester units and parents' qualifications are strongly correlated, violating the independence assumption. PCA to 15 components before Gaussian NB helped (MCC 0.434 to 0.474), but it still misses most Enrolled students (recall 0.20). |
| Support Vector Machine | Best AUC (0.880), recall (0.715) and MCC (0.602). A linear kernel with balanced weights won tuning, confirming the data is near-linearly separable. Enrolled recall 0.59, Dropout 0.75, Graduate 0.80. |
| **Overall winner** | **Support Vector Machine** by a narrow margin: highest AUC, recall and MCC (the best metrics for an imbalanced multi-class problem). Logistic Regression is virtually tied (it leads on accuracy, precision and F1 by under 0.003) and is the simpler choice. Balanced class weights traded about 1.5 points of raw accuracy for a large gain in minority-class recall. |
