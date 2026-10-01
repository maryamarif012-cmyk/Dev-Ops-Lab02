import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
#Data loading
df = pd.read_csv("diabetic_data.csv")
print("Dataset loaded:", df.shape)
y = (df["readmitted"] == "<30").astype(int)

X = df.drop(
    columns=[
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "readmitted"
    ],
    errors="ignore"
)

X = X.replace("?", "Unknown")
X = X.fillna("Unknown")
age_map = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95
}
X["age"] = X["age"].map(age_map)
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
# Identify categorical columns
cat_cols = X_train.select_dtypes(
    include=["object", "string"]
).columns
# One-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            cat_cols
        )
    ],
    remainder="passthrough"
)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
# Class imbalance
positive = y_train.sum()
negative = len(y_train) - positive
weight = negative / positive
# XGBoost
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)
print("\nTraining XGBoost...")
model.fit(X_train, y_train)
# Prediction
probability = model.predict_proba(X_test)[:, 1]
prediction = (probability >= 0.5).astype(int)
# Evaluation
print("       XGBoost Results")
print("ROC-AUC:", round(
    roc_auc_score(y_test, probability), 4
))
print("F1-Score:", round(
    f1_score(y_test, prediction), 4
))
print("Recall:", round(
    recall_score(y_test, prediction), 4
))
print("Precision:", round(
    precision_score(y_test, prediction), 4
))
print("\nXGBoost completed successfully.")