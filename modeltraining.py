
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

from xgboost import XGBClassifier

from config import RANDOM_STATE, RUN_SVM


def build_models():
    """
    Initialize the machine learning classifiers.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "KNN": KNeighborsClassifier(),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "AdaBoost": AdaBoostClassifier(
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1
        )
    }

    # SVM can be disabled through config.py
    if RUN_SVM:
        models["SVM"] = SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        )

    return models


def prepare_training_datasets(X_train, y_train):
    """
    Prepare original, SMOTE and SMOTE-ENN training datasets.

    Resampling is applied only to training data
    to prevent data leakage.
    """

    training_datasets = {}

    # 1. Original imbalanced dataset
    training_datasets["BASE"] = (
        X_train,
        y_train
    )

    print("\nApplying SMOTE...")

    # 2. SMOTE
    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    X_smote, y_smote = smote.fit_resample(
        X_train,
        y_train
    )

    training_datasets["SMOTE"] = (
        X_smote,
        y_smote
    )

    print("SMOTE completed.")

    print("\nApplying SMOTE-ENN...")

    # 3. SMOTE-ENN
    smote_enn = SMOTEENN(
        random_state=RANDOM_STATE,
        enn=EditedNearestNeighbours(
            sampling_strategy="majority"
        )
    )

    X_smoteenn, y_smoteenn = smote_enn.fit_resample(
        X_train,
        y_train
    )

    training_datasets["SMOTE-ENN"] = (
        X_smoteenn,
        y_smoteenn
    )

    print("SMOTE-ENN completed.")

    return training_datasets


def train_experiments(X_train, y_train):
    """
    Train all configured classifiers using
    BASE, SMOTE and SMOTE-ENN datasets.

    Returns a dictionary containing all trained models.
    """

    models = build_models()

    training_datasets = prepare_training_datasets(
        X_train,
        y_train
    )

    trained_models = {}

    print("\n========== MODEL TRAINING STARTED ==========")

    for sampling_name, (X_resampled, y_resampled) in training_datasets.items():

        print(f"\nTraining strategy: {sampling_name}")

        for model_name, model_template in models.items():

            print(f"Training {model_name}...")

            # Create a fresh model for each experiment
            model = clone(model_template)

            # Train model
            model.fit(
                X_resampled,
                y_resampled
            )

            # Store trained model
            trained_models[
                (sampling_name, model_name)
            ] = model

            print(f"Completed: {sampling_name} - {model_name}")

    print("\n========== MODEL TRAINING COMPLETED ==========")

    print(f"Total trained models: {len(trained_models)}")

    return trained_models


def display_training_summary(trained_models):
    """
    Display the list of successfully trained models.
    """

    summary = []

    for (sampling_name, model_name) in trained_models.keys():

        summary.append({
            "Sampling Strategy": sampling_name,
            "Model": model_name,
            "Status": "Trained"
        })

    summary_df = pd.DataFrame(summary)

    print("\nMODEL TRAINING SUMMARY")
    print(summary_df.to_string(index=False))

    return summary_df
