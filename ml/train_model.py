
import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours


# ============================================================
# 1. CONFIGURATION AND PATHS
# ============================================================

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "HR.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ML_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(MODEL_DIR, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.3f}".format)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset shape:", df.shape)
print("\nDataset information:")
df.info()

print("\nDuplicate rows:", df.duplicated().sum())

# IMPORTANT:
# Original Kaggle notebook does not remove duplicates.
# All duplicate rows are retained.


# ============================================================
# 3. HANDLE MISSING VALUES
# ORIGINAL NOTEBOOK LOGIC
# ============================================================

df["satisfaction_level"] = df["satisfaction_level"].fillna(
    df["satisfaction_level"].median()
)


# ============================================================
# 4. CATEGORICAL ENCODING
# ORIGINAL NOTEBOOK: pd.get_dummies()
# ============================================================

df = pd.concat(
    [
        df,
        pd.get_dummies(df["department"], prefix="department")
    ],
    axis=1
).drop(columns=["department"])


df = pd.concat(
    [
        df,
        pd.get_dummies(df["salary"], prefix="salary")
    ],
    axis=1
).drop(columns=["salary"])


print("\nDataset after categorical encoding:")
print(df.head())


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(["left"], axis=1)
y = df["left"]

feature_columns = X.columns.tolist()

print("\nFeature shape:", X.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 6. TRAIN-TEST SPLIT
# ORIGINAL NOTEBOOK PARAMETERS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=101
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# 7. FEATURE SCALING
# ORIGINAL NOTEBOOK: StandardScaler
# ============================================================

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Convert target Series into NumPy arrays.
# This is only a compatibility fix for newer pandas versions.
# It does not change the target values or training methodology.

y_train_array = np.asarray(y_train).ravel()
y_test_array = np.asarray(y_test).ravel()


print("\nPreprocessing completed successfully!")

print("\nTraining target distribution:")
print(pd.Series(y_train_array).value_counts())


# ============================================================
# 8. DEFINE ORIGINAL SEVEN CLASSIFIERS
# ORIGINAL NOTEBOOK DEFAULT PARAMETERS
# ============================================================

def create_models():

    return {
        "Logistic Regression": LogisticRegression(),

        "KNN": KNeighborsClassifier(),

        "SVM": SVC(
            kernel="rbf",
            probability=True
        ),

        "Decision Tree": DecisionTreeClassifier(),

        "Random Forest": RandomForestClassifier(),

        "AdaBoost": AdaBoostClassifier(),

        "XGBoost": XGBClassifier()
    }


model_names = [
    "Logistic Regression",
    "KNN",
    "SVM",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "XGBoost"
]


# ============================================================
# 9. STORAGE FOR EXPERIMENT RESULTS
# ============================================================

all_results = []

confusion_matrices = {}

roc_curve_data = {}

trained_models = {}


# ============================================================
# 10. MODEL EVALUATION FUNCTION
# ORIGINAL NOTEBOOK METRICS
# ============================================================

def evaluate_model(model, model_name, sampling_method):

    # Predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test_array,
        predictions
    )

    # Precision
    precision = precision_score(
        y_test_array,
        predictions
    )

    # Recall
    recall = recall_score(
        y_test_array,
        predictions
    )

    # F1 Score
    # Same formula used in original Kaggle notebook
    f1 = 2 * (precision * recall) / (precision + recall)

    # AUC Score
    probabilities = model.predict_proba(X_test)

    probabilities = probabilities[:, 1]

    auc = roc_auc_score(
        y_test_array,
        probabilities
    )

    experiment_name = f"{sampling_method} - {model_name}"

    # Store metrics
    result = {
        "Sampling": sampling_method,
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    }

    all_results.append(result)

    # Confusion Matrix
    cm = confusion_matrix(
        y_test_array,
        predictions
    )

    confusion_matrices[experiment_name] = cm.tolist()

    # Save trained model
    trained_models[experiment_name] = model

    # Save ROC coordinates for BASE models only.
    # Original notebook compares ROC curves for BASE models.

    if sampling_method == "BASE":

        fpr, tpr, thresholds = roc_curve(
            y_test_array,
            probabilities
        )

        roc_curve_data[model_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc)
        }

    # Display results
    print("\n----------------------------------------")
    print(f"{model_name} : {sampling_method}")
    print("----------------------------------------")

    print("Accuracy :", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall   :", round(recall, 3))
    print("F1 Score :", round(f1, 3))
    print("AUC Score:", round(auc, 3))

    print("\nConfusion Matrix:")
    print(cm)


# ============================================================
# 11. BASE EXPERIMENT
# ORIGINAL IMBALANCED DATA
# ============================================================

print("\n\n========================================")
print("BASE EXPERIMENT")
print("========================================")

base_models = create_models()

for model_name, model in base_models.items():

    model.fit(
        X_train,
        y_train_array
    )

    evaluate_model(
        model,
        model_name,
        "BASE"
    )


# ============================================================
# 12. SMOTE
# ORIGINAL NOTEBOOK: SMOTE()
# ============================================================

print("\n\n========================================")
print("SMOTE EXPERIMENT")
print("========================================")

smote = SMOTE()

smote_X_train, smote_y_train = smote.fit_resample(
    X_train,
    y_train
)

# Compatibility conversion
smote_y_train_array = np.asarray(smote_y_train).ravel()

print("\nSMOTE target distribution:")
print(pd.Series(smote_y_train_array).value_counts())


smote_models = create_models()

for model_name, model in smote_models.items():

    model.fit(
        smote_X_train,
        smote_y_train_array
    )

    evaluate_model(
        model,
        model_name,
        "SMOTE"
    )


# ============================================================
# 13. SMOTE-ENN
# ORIGINAL NOTEBOOK IMPLEMENTATION
# ============================================================

print("\n\n========================================")
print("SMOTE-ENN EXPERIMENT")
print("========================================")

smoteenn = SMOTEENN(
    enn=EditedNearestNeighbours(
        sampling_strategy="majority"
    )
)

smoteenn_X_train, smoteenn_y_train = smoteenn.fit_resample(
    X_train,
    y_train
)

# Compatibility conversion
smoteenn_y_train_array = np.asarray(smoteenn_y_train).ravel()

print("\nSMOTE-ENN target distribution:")
print(pd.Series(smoteenn_y_train_array).value_counts())


smoteenn_models = create_models()

for model_name, model in smoteenn_models.items():

    model.fit(
        smoteenn_X_train,
        smoteenn_y_train_array
    )

    evaluate_model(
        model,
        model_name,
        "SMOTE-ENN"
    )


# ============================================================
# 14. COMPLETE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(all_results)

print("\n\n========================================")
print("ALL 21 EXPERIMENT RESULTS")
print("========================================")

print(results_df.to_string(index=False))


# ============================================================
# 15. COMPARISON TABLES
# SAME ORIGINAL NOTEBOOK STRUCTURE
# ============================================================

sampling_methods = [
    "BASE",
    "SMOTE",
    "SMOTE-ENN"
]


accuracy_df = results_df.pivot(
    index="Sampling",
    columns="Model",
    values="Accuracy"
).reindex(
    index=sampling_methods,
    columns=model_names
)


precision_df = results_df.pivot(
    index="Sampling",
    columns="Model",
    values="Precision"
).reindex(
    index=sampling_methods,
    columns=model_names
)


recall_df = results_df.pivot(
    index="Sampling",
    columns="Model",
    values="Recall"
).reindex(
    index=sampling_methods,
    columns=model_names
)


f1_df = results_df.pivot(
    index="Sampling",
    columns="Model",
    values="F1"
).reindex(
    index=sampling_methods,
    columns=model_names
)


auc_df = results_df.pivot(
    index="Sampling",
    columns="Model",
    values="AUC"
).reindex(
    index=sampling_methods,
    columns=model_names
)


print("\nACCURACY")
print(accuracy_df)

print("\nRECALL")
print(recall_df)

print("\nF1 SCORE")
print(f1_df)

print("\nAUC")
print(auc_df)


# ============================================================
# 16. SAVE COMPARISON TABLES
# ============================================================

accuracy_df.to_csv(
    os.path.join(ML_DIR, "accuracy_comparison.csv")
)

precision_df.to_csv(
    os.path.join(ML_DIR, "precision_comparison.csv")
)

recall_df.to_csv(
    os.path.join(ML_DIR, "recall_comparison.csv")
)

f1_df.to_csv(
    os.path.join(ML_DIR, "f1_comparison.csv")
)

auc_df.to_csv(
    os.path.join(ML_DIR, "auc_comparison.csv")
)

results_df.to_csv(
    os.path.join(ML_DIR, "model_comparison.csv"),
    index=False
)


# ============================================================
# 17. SAVE CONFUSION MATRICES
# ============================================================

with open(
    os.path.join(MODEL_DIR, "confusion_matrices.json"),
    "w"
) as file:

    json.dump(
        confusion_matrices,
        file,
        indent=4
    )


# ============================================================
# 18. SAVE ROC CURVE DATA
# BASE MODELS ONLY
# ============================================================

with open(
    os.path.join(MODEL_DIR, "roc_curves.json"),
    "w"
) as file:

    json.dump(
        roc_curve_data,
        file,
        indent=4
    )


# ============================================================
# 19. SAVE EXPERIMENT RESULTS FOR FRONTEND
# ============================================================

with open(
    os.path.join(MODEL_DIR, "experiment_results.json"),
    "w"
) as file:

    json.dump(
        results_df.to_dict(orient="records"),
        file,
        indent=4
    )


# ============================================================
# 20. PLOT ROC CURVES
# SAME SEVEN BASE MODELS AS ORIGINAL NOTEBOOK
# ============================================================

ns_probs = [
    0 for _ in range(len(y_test_array))
]

ns_fpr, ns_tpr, _ = roc_curve(
    y_test_array,
    ns_probs
)

plt.figure(figsize=(8, 6))

plt.plot(
    ns_fpr,
    ns_tpr,
    linestyle="--",
    label="Based on Majority"
)

for model_name in model_names:

    curve = roc_curve_data[model_name]

    plt.plot(
        curve["fpr"],
        curve["tpr"],
        marker=".",
        label=model_name
    )

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    os.path.join(MODEL_DIR, "roc_curve_base.png"),
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. SAVE ALL 21 TRAINED MODELS
# ADDITIONAL EXPORT FOR APPLICATION INTEGRATION
# ============================================================

model_bundle = {
    "models": trained_models,
    "scaler": scaler,
    "feature_columns": feature_columns
}

joblib.dump(
    model_bundle,
    os.path.join(MODEL_DIR, "all_experiment_models.pkl")
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("========================================")

print("Original dataset rows:", len(df))

print("Duplicate rows retained:", df.duplicated().sum())

print("Training samples:", len(X_train))

print("Testing samples:", len(X_test))

print("Total experiments:", len(results_df))

print("\nGenerated files:")

print("ml/model_comparison.csv")
print("ml/accuracy_comparison.csv")
print("ml/precision_comparison.csv")
print("ml/recall_comparison.csv")
print("ml/f1_comparison.csv")
print("ml/auc_comparison.csv")

print("models/confusion_matrices.json")
print("models/roc_curves.json")
print("models/experiment_results.json")
print("models/roc_curve_base.png")
print("models/all_experiment_models.pkl")

print("\nAll 21 experiments evaluated.")
print("No best-model selection performed.")
