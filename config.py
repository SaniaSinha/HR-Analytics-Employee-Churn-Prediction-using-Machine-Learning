"""
Configuration and global environment settings for the
HR Analytics: Employee Churn Prediction project.
"""
import warnings
import pandas as pd
import seaborn as sns


def configure_environment():
    """Apply the same pandas/seaborn/warnings settings used in the original notebook."""
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    sns.set(color_codes=True)
    warnings.filterwarnings('ignore')


# Paths and constants
DATA_PATH = "data/HR.csv"
TEST_SIZE = 0.30
RANDOM_STATE = 101
