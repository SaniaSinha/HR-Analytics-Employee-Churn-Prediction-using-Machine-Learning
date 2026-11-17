
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATASET_CANDIDATES,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)


def load_dataset():
    """
    Load the HR employee dataset from the configured location.
    """

    dataset_path = next(
        (path for path in DATASET_CANDIDATES if path.exists()),
        None
    )

    if dataset_path is None:
        searched_paths = "\n".join(
            str(path) for path in DATASET_CANDIDATES
        )
        raise FileNotFoundError(
            f"HR.csv not found. Expected it in one of:\n{searched_paths}"
        )

    df = pd.read_csv(dataset_path)

    if df.empty:
        raise ValueError("The uploaded HR dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset."
        )

    print("Dataset loaded successfully!")
    print(f"Dataset location: {dataset_path}")
    print(f"Dataset shape: {df.shape}")

    return df


def preprocess_data(df):
    """
    Clean the dataset and encode categorical variables.
    """

    df = df.copy()

    # Handle missing values in satisfaction_level
    if "satisfaction_level" in df.columns:
        df["satisfaction_level"] = df[
            "satisfaction_level"
        ].fillna(df["satisfaction_level"].median())

    # Handle any remaining missing numerical values
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.drop(TARGET_COLUMN, errors="ignore")

    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    # Handle missing categorical values
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        if df[column].isnull().any():
            mode_values = df[column].mode()

            if not mode_values.empty:
                df[column] = df[column].fillna(mode_values[0])

    # Check for remaining missing values
    if df.isnull().sum().any():
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        raise ValueError(
            f"Missing values remain:\n{missing}"
        )

    # Encode categorical features
    categorical_features = [
        column
        for column in ["department", "salary"]
        if column in df.columns
    ]

    if categorical_features:
        df = pd.get_dummies(
            df,
            columns=categorical_features,
            dtype=int
        )

    print("Data preprocessing completed!")
    print(f"Processed dataset shape: {df.shape}")

    return df


def split_and_scale_data(df):
    """
    Separate features and target, split the dataset,
    and apply feature scaling.
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if y.nunique() != 2:
        raise ValueError(
            f"Expected a binary target column. "
            f"Found {y.nunique()} unique values."
        )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Train-test split completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X.shape[1]}")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        X.columns.tolist()
    )


def prepare_data():
    """
    Execute the complete preprocessing pipeline.
    """

    df = load_dataset()

    df = preprocess_data(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = split_and_scale_data(df)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names
    }
