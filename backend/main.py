import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.database import client, predictions_collection
from backend.schemas.employee import EmployeeData


# ============================================================
# 1. PATHS AND APPLICATION SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_BUNDLE_PATH = MODEL_DIR / "all_experiment_models.pkl"
RESULTS_PATH = MODEL_DIR / "experiment_results.json"
CONFUSION_PATH = MODEL_DIR / "confusion_matrices.json"
ROC_PATH = MODEL_DIR / "roc_curves.json"

app = FastAPI(
    title="HR Employee Churn Analytics API",
    description="Employee churn prediction and model evaluation system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# 2. LOAD EXPERIMENT MODELS
# ============================================================

if not MODEL_BUNDLE_PATH.exists():
    raise FileNotFoundError(
        "Model bundle not found. Run python ml/train_model.py first."
    )

model_bundle = joblib.load(MODEL_BUNDLE_PATH)

experiment_models = model_bundle["models"]
scaler = model_bundle["scaler"]
feature_columns = model_bundle["feature_columns"]


# ============================================================
# 3. LOAD EVALUATION RESULTS
# ============================================================

def load_json_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required evaluation file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


experiment_results = load_json_file(RESULTS_PATH)
confusion_matrices = load_json_file(CONFUSION_PATH)
roc_curves = load_json_file(ROC_PATH)


# ============================================================
# 4. AUTOMATICALLY SELECT BEST EXPERIMENT USING F1
# ============================================================

if not isinstance(experiment_results, list) or not experiment_results:
    raise ValueError(
        "Experiment results must be a non-empty list."
    )

best_experiment = max(
    experiment_results,
    key=lambda result: float(result["F1"])
)

BEST_EXPERIMENT_NAME = (
    f'{best_experiment["Sampling"]} - {best_experiment["Model"]}'
)

BEST_F1_SCORE = float(best_experiment["F1"])

if BEST_EXPERIMENT_NAME not in experiment_models:
    raise ValueError(
        f"Best experiment model not found: {BEST_EXPERIMENT_NAME}"
    )

best_model = experiment_models[BEST_EXPERIMENT_NAME]

print("\n========== AUTOMATIC MODEL SELECTION ==========")
print(f"Selected experiment : {BEST_EXPERIMENT_NAME}")
print(f"F1-score            : {BEST_F1_SCORE:.4f}")
print("===============================================\n")


# ============================================================
# 5. DATABASE HELPER FUNCTIONS
# ============================================================

def is_mongodb_available():

    if client is None or predictions_collection is None:
        return False

    try:
        client.admin.command("ping")
        return True
    except Exception as error:
        print(f"MongoDB unavailable: {error}")
        return False


def save_prediction(document):

    if not is_mongodb_available():
        print("Prediction returned without MongoDB storage.")
        return False

    try:
        predictions_collection.insert_one(document)
        return True
    except Exception as error:
        print(f"MongoDB save failed: {error}")
        return False


# ============================================================
# 6. ROOT AND HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def home():

    return {
        "message": "HR Employee Churn Analytics API",
        "status": "running",
        "experiments_available": len(experiment_models),
        "selected_experiment": BEST_EXPERIMENT_NAME,
        "selected_f1_score": BEST_F1_SCORE
    }


@app.get("/health")
def health():

    mongo_status = (
        "connected"
        if is_mongodb_available()
        else "disconnected"
    )

    return {
        "api": "running",
        "model_bundle": "loaded",
        "experiments_available": len(experiment_models),
        "selected_experiment": BEST_EXPERIMENT_NAME,
        "selected_f1_score": BEST_F1_SCORE,
        "mongo": mongo_status
    }


# ============================================================
# 7. EVALUATION ENDPOINT
# ============================================================

@app.get("/evaluation")
def get_evaluation():

    return {
        "results": experiment_results,
        "confusion_matrices": confusion_matrices,
        "roc_curves": roc_curves,
        "best_experiment": BEST_EXPERIMENT_NAME,
        "best_f1_score": BEST_F1_SCORE
    }


# ============================================================
# 8. PREPARE EMPLOYEE INPUT
# SAME ENCODING AND SCALING AS TRAINING
# ============================================================

def prepare_employee_data(employee_data):

    input_df = pd.DataFrame([employee_data])

    input_df = pd.get_dummies(
        input_df,
        columns=["department", "salary"],
        prefix=["department", "salary"]
    )

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    scaled_input = scaler.transform(input_df)

    return scaled_input


# ============================================================
# 9. PREDICTION ENDPOINT
# AUTOMATICALLY USES THE HIGHEST F1 EXPERIMENT
# ============================================================

@app.post("/predict")
def predict_churn(request: EmployeeData):

    try:

        # Get employee details
        employee_data = request.model_dump()

        # Prepare input using original training preprocessing
        processed_data = prepare_employee_data(employee_data)

        # Use automatically selected best model
        prediction = int(
            best_model.predict(processed_data)[0]
        )

        probabilities = best_model.predict_proba(
            processed_data
        )[0]

        positive_class_index = list(
            best_model.classes_
        ).index(1)

        churn_probability = float(
            probabilities[positive_class_index]
        )

        prediction_label = (
            "Likely to Leave"
            if prediction == 1
            else "Likely to Stay"
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        response = {
            "experiment": BEST_EXPERIMENT_NAME,
            "f1_score": round(BEST_F1_SCORE, 6),
            "prediction": prediction,
            "prediction_label": prediction_label,
            "probability": round(churn_probability, 4),
            "churn_probability": round(churn_probability, 4),
            "timestamp": timestamp,
            "database_saved": False
        }

        document = {
            **employee_data,
            **response
        }

        # MongoDB failure must never block prediction
        response["database_saved"] = save_prediction(document)

        return response

    except HTTPException:
        raise

    except Exception as error:

        print(f"Prediction error: {error}")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# ============================================================
# 10. DASHBOARD ENDPOINT
# ============================================================

@app.get("/dashboard")
def dashboard():

    if not is_mongodb_available():

        return {
            "total_predictions": 0,
            "at_risk": 0,
            "likely_to_stay": 0,
            "churn_rate": 0,
            "recent_predictions": [],
            "database_status": "disconnected"
        }

    try:

        total = predictions_collection.count_documents({})

        at_risk = predictions_collection.count_documents({
            "prediction": 1
        })

        likely_to_stay = predictions_collection.count_documents({
            "prediction": 0
        })

        churn_rate = (
            (at_risk / total) * 100
            if total > 0
            else 0
        )

        recent = list(
            predictions_collection.find(
                {},
                {"_id": 0}
            ).sort(
                "timestamp",
                -1
            ).limit(5)
        )

        return {
            "total_predictions": total,
            "at_risk": at_risk,
            "likely_to_stay": likely_to_stay,
            "churn_rate": round(churn_rate, 2),
            "recent_predictions": recent,
            "database_status": "connected"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Dashboard error: {str(error)}"
        )


# ============================================================
# 11. PREDICTION HISTORY ENDPOINT
# ============================================================

@app.get("/history")
def prediction_history():

    if not is_mongodb_available():

        return {
            "total_records": 0,
            "history": [],
            "database_status": "disconnected"
        }

    try:

        records = list(
            predictions_collection.find(
                {},
                {"_id": 0}
            ).sort(
                "timestamp",
                -1
            ).limit(100)
        )

        return {
            "total_records": len(records),
            "history": records,
            "database_status": "connected"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"History error: {str(error)}"
        )
