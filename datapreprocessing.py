"""
Data loading, exploratory checks, cleaning, categorical encoding,
and train/test split + scaling utilities.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE


def load_data(path=DATA_PATH):
    """Load the HR dataset from CSV."""
    df = pd.read_csv(path)
    return df


def explore_data(df):
    """Run the same exploratory checks as the original notebook."""
    print(df.head())
    print("Shape:", df.shape)
    print(df.describe())
    df.info()

    # Distribution of satisfaction_level
    df["satisfaction_level"].hist()
    plt.show()

    print(df['department'].value_counts())
    print(df['salary'].value_counts())


def clean_data(df):
    """Impute missing values in 'satisfaction_level' with the median."""
    df["satisfaction_level"] = df["satisfaction_level"].fillna(df["satisfaction_level"].median())
    return df


def encode_categorical(df):
    """Convert 'department' and 'salary' into one-hot dummy variables."""
    # Convert "department" into dummy variables
    df = pd.concat(
        [df, pd.get_dummies(df["department"], prefix="department")], axis=1
    ).drop(columns=["department"])

    # Convert "salary" into dummy variables
    df = pd.concat(
        [df, pd.get_dummies(df["salary"], prefix="salary")], axis=1
    ).drop(columns=["salary"])

    # Display the part of the train set to see if the categorical variables were properly converted.
    print(df.head(10))

    return df


def split_and_scale(df, target="left", test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split into train/test sets and standardize features."""
    X = df.drop([target], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def check_data_balance(y):
    """Print the class counts of the target (e.g. churn = 1 vs 0) to check imbalance."""
    print(y.value_counts())


def churn_rate(df, target="left"):
    """Calculate the overall churn (attrition) rate as a percentage."""
    return (sum(df[target]) / len(df[target].index)) * 100
