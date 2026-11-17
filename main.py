
from datapreprocessing import prepare_data
from modeltraining import train_experiments, display_training_summary
from evaluate import (
    evaluate_models,
    plot_roc_curves,
    display_best_model
)


def main():
    print("\n" + "=" * 60)
    print("HR ANALYTICS - EMPLOYEE CHURN PREDICTION")
    print("=" * 60)

    # Step 1: Load and preprocess the dataset
    print("\n[1/4] Loading and preprocessing dataset...")

    data = prepare_data()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("Dataset preprocessing completed successfully.")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")

    # Step 2: Train models
    print("\n[2/4] Training machine learning models...")

    trained_models = train_experiments(X_train, y_train)

    display_training_summary(trained_models)

    # Step 3: Evaluate models
    print("\n[3/4] Evaluating model performance...")

    results_df = evaluate_models(
        trained_models,
        X_test,
        y_test
    )

    # Step 4: Generate visualizations and results
    print("\n[4/4] Generating ROC curves and final results...")

    plot_roc_curves(
        trained_models,
        X_test,
        y_test
    )

    display_best_model(results_df)

    print("\n" + "=" * 60)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nCheck the 'outputs' folder for evaluation results and plots.")


if __name__ == "__main__":
    main()
