# HR-Analytics-Employee-Churn-Prediction-using-Machine-Learning

Predicts employee attrition using multiple ML models and compares strategies for handling class imbalance.

Features
7 classifiers trained and compared (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, AdaBoost, XGBoost)
Baseline vs SMOTE vs SMOTE-ENN imbalance handling
Accuracy, Precision, Recall, F1, AUC comparison tables
ROC curve visualization
One-hot encoding for department and salary
Modular pipeline (config, preprocessing, training, evaluation)
Install dependencies
pip install -r requirements.txt
Add dataset

Place HR.csv in the data/ folder.

Run full pipeline
python main.py
Project Structure
src/config.py            - global settings
src/data_preprocessing.py - load, clean, encode, split, scale
src/model_training.py     - models, training, SMOTE, SMOTE-ENN
src/evaluate.py           - results tables, ROC curves
main.py                   - runs everything
