# HR Analytics: Employee Churn Prediction Using Machine Learning

## Overview

This project uses machine learning techniques to analyze employee data and predict employee attrition.

It explores different classification algorithms and compares their performance using multiple evaluation metrics.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Seaborn

## Machine Learning Models

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost
- Support Vector Machine (SVM)

## Class Imbalance Handling

Three training strategies are compared:

- Baseline
- SMOTE
- SMOTE-ENN

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curves

## Project Structure

```text
HR-Analytics-Employee-Churn-Prediction/
│
├── config.py
├── datapreprocessing.py
├── modeltraining.py
├── evaluate.py
├── main.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── HR.csv
│
└── outputs/
```

## Setup and Execution

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
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the dataset

Place `HR.csv` inside the `data/` folder or the project root directory.

The dataset is not included in this repository.

### 6. Run the project

```bash
python main.py
```

## Author

**Sania Sinha**

MCA – Data Science, NIT Patna

[GitHub Profile](https://github.com/SaniaSinha)
