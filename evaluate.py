"""
Aggregate model results into comparison tables (Accuracy, Recall, F1, AUC)
across the three imbalance-handling strategies, and plot ROC curves.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

MODEL_NAMES = [
    'Logistic Regression', 'KNN', 'SVM', 'Decision Tree',
    'Random Forest', 'AdaBoost', 'XGBoost'
]


def build_results_table(base_results, smote_results, smoteenn_results, metric):
    """Build a BASE / SMOTE / SMOTE-ENN x model comparison table for one metric."""
    pd.options.display.float_format = '{:.3f}'.format

    data = [
        [base_results[name][metric] for name in MODEL_NAMES],
        [smote_results[name][metric] for name in MODEL_NAMES],
        [smoteenn_results[name][metric] for name in MODEL_NAMES],
    ]

    df = pd.DataFrame(data)
    df.index = ['BASE', 'SMOTE', 'SMOTE-ENN']
    df.columns = MODEL_NAMES
    return df


def build_all_results_tables(base_results, smote_results, smoteenn_results):
    """Build Accuracy, Recall, F1, and AUC comparison tables."""
    tables = {}
    for metric in ["accuracy", "recall", "f1", "auc"]:
        tables[metric] = build_results_table(base_results, smote_results, smoteenn_results, metric)
    return tables


def plot_roc_curves(base_results, y_test):
    """Plot ROC curves for all baseline (imbalanced-data) models."""
    ns_probs = [0 for _ in range(len(y_test))]
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Based on Majority')

    for name in MODEL_NAMES:
        probs = base_results[name]["probs"]
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, marker='.', label=name)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
