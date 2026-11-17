
import os
from pathlib import Path

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent

# Project directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ==========================================
# DATASET CONFIGURATION
# ==========================================

# HR.csv is private and should not be uploaded
# to the public GitHub repository.

DATASET_CANDIDATES = [
    PROJECT_ROOT / "HR.csv",
    DATA_DIR / "HR.csv"
]

TARGET_COLUMN = "left"

# ==========================================
# MACHINE LEARNING CONFIGURATION
# ==========================================

TEST_SIZE = 0.30
RANDOM_STATE = 101

RUN_SVM = True

# ==========================================
# FUTURE FULL-STACK CONFIGURATION
# ==========================================

APP_NAME = "HR Analytics Employee Churn Prediction"

# MongoDB settings are loaded from environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "hr_analytics")

# ==========================================
# OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
