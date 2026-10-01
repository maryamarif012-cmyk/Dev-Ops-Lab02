import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
df = pd.read_csv("diabetic_data.csv")
print("Dataset loaded:", df.shape)
y = (df["readmitted"] == "<30").astype(int)

X = df.drop(
    ["encounter_id", "patient_nbr", "weight",
     "payer_code", "readmitted"],
    axis=1
)
# 4. Handle missing values
X = X.replace("?", "Unknown")
X = X.fillna("Unknown")
# 5. Convert age into numbers
age_values = {
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
X["age"] = X["age"].map(age_values)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns
preprocessor = ColumnTransformer(
    [
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print("Preprocessing completed.")
positive = y_train.sum()
negative = len(y_train) - positive

class_weight = negative / positive
print("Positive class weight:", round(class_weight, 2))
print("\nTraining XGBoost...")

xgb = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=class_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

print("XGBoost completed.")
print("\nTraining LightGBM...")
lgb = LGBMClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb.fit(X_train, y_train)
print("LightGBM completed.")
print("\nTraining Logistic Regression...")
lr = LogisticRegression(
    class_weight="balanced",
    max_iter=500,
    solver="liblinear"
)
lr.fit(X_train, y_train)
print("Logistic Regression completed.")
xgb_train = xgb.predict_proba(X_train)[:, 1]
lgb_train = lgb.predict_proba(X_train)[:, 1]
lr_train = lr.predict_proba(X_train)[:, 1]
xgb_test = xgb.predict_proba(X_test)[:, 1]
lgb_test = lgb.predict_proba(X_test)[:, 1]
lr_test = lr.predict_proba(X_test)[:, 1]
stack_train = np.column_stack(
    [xgb_train, lgb_train, lr_train]
)

stack_test = np.column_stack(
    [xgb_test, lgb_test, lr_test]
)
print("\nTraining Stacking Meta-Learner...")
meta_model = LogisticRegression(
    max_iter=500,
    solver="liblinear"
)
meta_model.fit(
    stack_train,
    y_train
)
print("Meta-learner completed.")
final_probability = meta_model.predict_proba(
    stack_test
)[:, 1]
final_prediction = (
    final_probability >= 0.5
).astype(int)
roc_auc = roc_auc_score(
    y_test,
    final_probability
)
f1 = f1_score(
    y_test,
    final_prediction
)
recall = recall_score(
    y_test,
    final_prediction
)
precision = precision_score(
    y_test,
    final_prediction
)
print("       STACKING RESULTS")
print("ROC-AUC:   ", round(roc_auc, 4))
print("F1-Score:  ", round(f1, 4))
print("Recall:    ", round(recall, 4))
print("Precision: ", round(precision, 4))

print("\nStacking completed successfully.")