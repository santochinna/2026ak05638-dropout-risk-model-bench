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
- **Preprocessing:** header clean-up, `StandardScaler` inside every pipeline (so scaling is saved with each model), target label-encoded as Dropout=0, Enrolled=1, Graduate=2.
- **Tuning:** 5-fold stratified `GridSearchCV` on the training set, scored by macro-F1.
- **Metric averaging:** Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro; MCC is the multi-class version.

## c. GitHub repository link
https://github.com/santochinna/2026ak05638-dropout-risk-model-bench

Live app: _<add Streamlit Community Cloud link>_

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
| Logistic Regression | 0.7525 | 0.8716 | 0.6783 | 0.6501 | 0.6531 | 0.5876 |
| Decision Tree | 0.7367 | 0.8428 | 0.6935 | 0.6711 | 0.6783 | 0.5684 |
| KNN | 0.7085 | 0.7906 | 0.6507 | 0.5994 | 0.6046 | 0.5104 |
| Naive Bayes | 0.6633 | 0.7751 | 0.5728 | 0.5612 | 0.5603 | 0.4339 |
| Support Vector Machine | 0.7627 | 0.8723 | 0.7055 | 0.6678 | 0.6768 | 0.6054 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, stable baseline (accuracy 0.753, AUC 0.872). Student features are largely linear signals (e.g. approved units), so it separates Graduate (recall 0.92) and Dropout (0.77) well but misses most Enrolled students (recall 0.26). |
| Decision Tree | Best at the hard minority class (Enrolled recall 0.46), giving the top macro recall/F1 among all models, but lower AUC (0.843) and accuracy (0.737) because hard splits give coarse probabilities. Tuned depth of 4 keeps it from overfitting and easy to interpret. |
| KNN | Middle-of-the-pack (accuracy 0.709, AUC 0.791). With 36 mixed features, distances become less informative; many categorical codes scaled as numbers add noise. Enrolled recall is only 0.21. |
| Naive Bayes | Weakest model (accuracy 0.663, MCC 0.434). The independence assumption is badly violated: 1st and 2nd-semester units and parents' qualifications are strongly correlated. It still gets a reasonable AUC (0.775), so its ranking is better than its hard predictions. |
| Support Vector Machine | Best on accuracy (0.763), AUC (0.872), precision (0.706) and MCC (0.605). The linear kernel won tuning, confirming the data is near-linearly separable. Like all models it struggles with Enrolled (recall 0.33). |
| **Overall winner** | **Support Vector Machine** - highest accuracy, AUC, precision and MCC, with macro-F1 (0.677) within 0.002 of the best (Decision Tree). Logistic Regression is a very close, simpler runner-up. The "Enrolled" class is the main weakness of every model and the best target for future work (class weighting, resampling). |
