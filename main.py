"""
HR Analytics: Employee Churn Prediction
=========================================
Entry point that runs the full pipeline: data loading & cleaning,
feature encoding, train/test split & scaling, model training under
three imbalance-handling strategies (baseline, SMOTE, SMOTE-ENN),
and final evaluation (comparison tables + ROC curves).

Run with:
    python main.py
"""
from src.config import configure_environment
from src.data_preprocessing import (
    load_data, explore_data, clean_data, encode_categorical, split_and_scale,
    check_data_balance, churn_rate
)
from src.model_training import run_all_models, apply_smote, apply_smoteenn
from src.evaluate import build_all_results_tables, plot_roc_curves


def main():
    configure_environment()

    # 1. Load and explore the data
    df = load_data()
    explore_data(df)

    # 2. Clean and encode categorical features
    df = clean_data(df)
    df = encode_categorical(df)

    print("Churn rate: {:.2f}%".format(churn_rate(df)))

    # 3. Split into train/test and scale
    X_train, X_test, y_train, y_test = split_and_scale(df)

    # Check the data balance (class counts of the target)
    check_data_balance(df["left"])

    # 4. Baseline models (imbalanced data)
    base_results = run_all_models(X_train, y_train, X_test, y_test, strategy="Imbalanced Data")

    # 5. SMOTE-resampled models
    smote_X_train, smote_y_train = apply_smote(X_train, y_train)
    smote_results = run_all_models(smote_X_train, smote_y_train, X_test, y_test, strategy="SMOTE")

    # 6. SMOTE-ENN resampled models
    smoteenn_X_train, smoteenn_y_train = apply_smoteenn(X_train, y_train)
    smoteenn_results = run_all_models(
        smoteenn_X_train, smoteenn_y_train, X_test, y_test, strategy="SMOTE-ENN"
    )

    # 7. Build and print comparison tables
    tables = build_all_results_tables(base_results, smote_results, smoteenn_results)
    for metric, table in tables.items():
        print(f"\n{metric.upper()}")
        print(table)

    # 8. ROC curves for baseline models
    plot_roc_curves(base_results, y_test)


if __name__ == "__main__":
    main()
