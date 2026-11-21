
# HR Analytics: Employee Churn Prediction Using Machine Learning

An end-to-end Machine Learning application that predicts employee attrition using HR data. The project compares multiple machine learning algorithms and sampling techniques to identify the best-performing model for employee churn prediction.

## Project Overview

Employee attrition is an important challenge for organizations. Identifying employees who are likely to leave can help HR teams understand workforce trends and make better retention decisions.

This project uses historical employee data to:
- Analyze employee characteristics and attrition patterns.
- Handle class imbalance using different sampling techniques.
- Train and evaluate multiple machine learning models.
- Automatically select the best-performing experiment based on F1-score.
- Predict employee churn through a web application.
- Store prediction records and maintain prediction history using MongoDB.

## Key Features

- **21 Machine Learning Experiments:** Seven algorithms evaluated under three sampling strategies.
- **Automatic Model Selection:** Selects the experiment with the highest F1-score.
- **Employee Churn Prediction:** Predicts whether an employee is likely to stay or leave.
- **Model Evaluation Dashboard:** Displays Accuracy, Precision, Recall, F1-score, ROC-AUC and confusion matrices.
- **ROC Curve Visualization:** Compares the seven baseline classifiers.
- **Prediction History:** Stores and displays previous predictions.
- **Interactive Dashboard:** Displays prediction statistics and recent activity.

## Machine Learning Methodology

### 1. Dataset

The project uses an HR employee dataset containing the following features:

- Satisfaction Level
- Last Evaluation
- Number of Projects
- Average Monthly Hours
- Time Spent at Company
- Work Accident
- Promotion in Last Five Years
- Department
- Salary
- Left (Target Variable)

### 2. Data Preprocessing

- Handled missing values using median imputation.
- Applied one-hot encoding to categorical features.
- Split the dataset into training and testing sets using stratified sampling.
- Applied StandardScaler for feature scaling.
- Preserved the original dataset records and duplicates.

### 3. Sampling Techniques

Three approaches were evaluated:

1. **BASE:** Original imbalanced training dataset.
2. **SMOTE:** Synthetic Minority Over-sampling Technique.
3. **SMOTE-ENN:** Combination of oversampling and Edited Nearest Neighbours.

### 4. Machine Learning Algorithms

Seven classification algorithms were trained:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree
5. Random Forest
6. AdaBoost
7. XGBoost

Combining seven algorithms with three sampling strategies resulted in **21 experiments**.

### 5. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The F1-score was used as the selection criterion for the final prediction model.

## Best Performing Experiment

Based on the evaluation results:

| Metric | Result |
|---|---:|
| Selected Experiment | BASE - Random Forest |
| Accuracy | 98.76% |
| Precision | 99.13% |
| Recall | 95.62% |
| F1-score | 97.34% |
| ROC-AUC | 99.19% |

The application automatically selects the experiment with the highest F1-score from the 21 evaluated experiments.

## Application Architecture

```text
HR Churn Prediction
│
├── Frontend
│   ├── Dashboard
│   ├── Predict Churn
│   ├── Model Evaluation
│   ├── History
│   └── About
│
├── Backend (FastAPI)
│   ├── Prediction API
│   ├── Evaluation API
│   ├── Dashboard API
│   └── History API
│
├── Machine Learning
│   ├── Data Preprocessing
│   ├── Model Training
│   ├── Model Evaluation
│   └── Automatic Model Selection
│
└── MongoDB
    └── Prediction Records
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalanced Data | Imbalanced-learn |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript, Bootstrap |
| Database | MongoDB |
| Model Persistence | Joblib |
| Development | PyCharm, Git, GitHub |

## Project Structure

```text
HR Churn project/
│
├── backend/
│   ├── main.py
│   ├── database.py
│   └── schemas/
│
├── data/
│   └── HR.csv
│
├── frontend/
│   ├── index.html
│   ├── predict.html
│   ├── evaluation.html
│   ├── history.html
│   ├── about.html
│   ├── css/
│   └── js/
│
├── ml/
│   └── train_model.py
│
├── models/
│   ├── all_experiment_models.pkl
│   ├── experiment_results.json
│   ├── confusion_matrices.json
│   └── roc_curves.json
│
├── .gitignore
└── requirements.txt
```

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/SaniaSinha/HR-Analytics-Employee-Churn-Prediction-using-Machine-Learning.git
```

### 2. Navigate to the project directory

```bash
cd HR-Analytics-Employee-Churn-Prediction-using-Machine-Learning
```

### 3. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure MongoDB

Create a `.env` file in the project root:

```env
MONGODB_URI=your_mongodb_connection_string
DATABASE_NAME=hr_churn_db
```

### 6. Start the FastAPI backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

Backend API:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

### 7. Open the frontend

Open `frontend/index.html` using a local development server, such as VS Code Live Server.

Make sure the backend is running before using the application.

## Future Improvements

- Model monitoring and performance tracking.
- Explainable AI using SHAP.
- Advanced employee attrition analytics.
- Additional HR insights and visualizations.

## Author

**Sania Sinha**

MCA – Data Science  
National Institute of Technology, Patna
