
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

from config import OUTPUT_DIR


def calculate_metrics(model, X_test, y_test):
    """
    Calculate classification performance metrics.
    """

    predictions = model.predict(X_test)

    # Obtain prediction probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]

    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X_test)

    else:
        probabilities = None

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    auc = (
        roc_auc_score(y_test, probabilities)
        if probabilities is not None
        else float("nan")
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    }, predictions, probabilities


def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate all trained models across all
    sampling strategies.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n========== MODEL EVALUATION STARTED ==========")

    for (sampling_name, model_name), model in trained_models.items():

        print(f"\nEvaluating: {sampling_name} - {model_name}")

        metrics, predictions, probabilities = calculate_metrics(
            model,
            X_test,
            y_test
        )

        # Store model performance
        result = {
            "Sampling Strategy": sampling_name,
            "Model": model_name,
            **metrics
        }

        results.append(result)

        # Save confusion matrix
        matrix = confusion_matrix(y_test, predictions)

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )

        safe_name = (
            f"{sampling_name}_{model_name}"
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        matrix_df.to_csv(
            OUTPUT_DIR / f"confusion_{safe_name}.csv"
        )

        # Save classification report
        report = classification_report(
            y_test,
            predictions,
            zero_division=0
        )

        report_path = OUTPUT_DIR / f"report_{safe_name}.txt"

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report)

    # Create comparison dataframe
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    # Save comparison results
    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n========== MODEL COMPARISON ==========")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\nEvaluation completed successfully!")

    return results_df


def plot_roc_curves(trained_models, X_test, y_test):
    """
    Plot ROC curves for baseline models.
    """

    plt.figure(figsize=(10, 7))

    plotted_models = 0

    for (sampling_name, model_name), model in trained_models.items():

        # Compare models trained on original data
        if sampling_name != "BASE":
            continue

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]

        elif hasattr(model, "decision_function"):
            probabilities = model.decision_function(X_test)

        else:
            continue

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {auc:.3f})"
        )

        plotted_models += 1

    if plotted_models == 0:
        plt.close()
        print("No baseline models available for ROC plotting.")
        return

    # Random classifier reference line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve - Baseline Models")

    plt.legend(loc="lower right")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save ROC curve
    roc_path = OUTPUT_DIR / "baseline_roc_curves.png"

    plt.savefig(
        roc_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nROC curve saved at: {roc_path}")


def display_best_model(results_df):
    """
    Display the best performing model based on F1 score.
    """

    if results_df.empty:
        print("No evaluation results available.")
        return

    best_model = results_df.iloc[0]

    print("\n========== BEST MODEL ==========")

    print(f"Sampling Strategy: {best_model['Sampling Strategy']}")
    print(f"Model: {best_model['Model']}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"Precision: {best_model['Precision']:.4f}")
    print(f"Recall: {best_model['Recall']:.4f}")
    print(f"F1 Score: {best_model['F1 Score']:.4f}")
    print(f"ROC-AUC: {best_model['ROC-AUC']:.4f}")

    return best_model.to_dict()
