"""
Model training utilities.

Includes:
- get_model_dict(): the 7 classifiers used in the original notebook
  (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, AdaBoost, XGBoost)
- train_and_evaluate(): fits one model and computes Accuracy/Precision/Recall/F1/AUC
  + confusion matrix (mirrors the repeated blocks in the original notebook)
- run_all_models(): runs train_and_evaluate() for every model in get_model_dict()
- apply_smote() / apply_smoteenn(): resampling strategies for the imbalanced target
"""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours


def get_model_dict():
    """Return a fresh dictionary of model name -> unfitted estimator."""
    return {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(kernel='rbf', probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost": XGBClassifier(),
    }


def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name="", strategy=""):
    """
    Fit a model and compute Accuracy, Precision, Recall, F1, AUC and the
    confusion matrix -- equivalent to each repeated block in the original notebook.
    """
    fitted_model = model.fit(X_train, y_train.ravel())
    pred = fitted_model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = 2 * (precision * recall) / (precision + recall)

    probs = fitted_model.predict_proba(X_test)
    probs = probs[:, 1]
    auc = roc_auc_score(y_test, probs)

    cm = confusion_matrix(y_test, pred)

    print(f"{model_name}: {strategy}")
    print(" - Accuracy : ", '{:.3f}'.format(accuracy))
    print(" - Precision: ", '{:.3f}'.format(precision))
    print(" - Recall   : ", '{:.3f}'.format(recall))
    print(" - F1 score : ", '{:.3f}'.format(f1))
    print(" - AUC score: ", '{:.3f}'.format(auc))
    print("Confusion Matrix")
    print(cm)

    return {
        "model": fitted_model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "probs": probs,
        "confusion_matrix": cm,
    }


def run_all_models(X_train, y_train, X_test, y_test, strategy="Imbalanced Data"):
    """Train every model in get_model_dict() and return results keyed by model name."""
    results = {}
    for name, model in get_model_dict().items():
        results[name] = train_and_evaluate(model, X_train, y_train, X_test, y_test, name, strategy)
    return results


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to the training set."""
    smote = SMOTE()
    smote_X_train, smote_y_train = smote.fit_resample(X_train, y_train)
    print(smote_y_train.value_counts())
    return smote_X_train, smote_y_train


def apply_smoteenn(X_train, y_train):
    """Apply combined SMOTE + Edited Nearest Neighbours resampling to the training set."""
    smoteenn_X_train, smoteenn_y_train = SMOTEENN(
        enn=EditedNearestNeighbours(sampling_strategy='majority')
    ).fit_resample(X_train, y_train)
    print(smoteenn_y_train.value_counts())
    return smoteenn_X_train, smoteenn_y_train
