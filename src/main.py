import os
import pandas as pd
from model import model_training
from evaluation import evaluate_model
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Load the dataset (No need for train-test split now)
data_dir = os.path.abspath("../ABC/dataset/processed")

X = pd.read_csv(os.path.join(data_dir, "X.csv"))
y = pd.read_csv(os.path.join(data_dir, "y.csv")).values.ravel()  # Convert to 1D array

#model directory
model_paths = {
    "logistic regression": os.path.abspath(os.path.join("../ABC", "models", "logistic_regression.pkl")),
    "random forest": os.path.abspath(os.path.join("../ABC", "models", "random_forest.pkl")),
    "XGBoost": os.path.abspath(os.path.join("../ABC", "models", "XGBoost.pkl")),
    "SVM": os.path.abspath(os.path.join("../ABC", "models", "SVM.pkl"))
}

# Cross-validation folds
cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Iterate over models, train, save, and evaluate
for model_name, model_path in model_paths.items():
    # Train the model with cross-validation
    model, scaler, mean_auc = model_training(X, y, model_name, model_path, cv=cv_folds)

    # Evaluate model using cross-validation and log the metrics
    log_path = os.path.abspath(os.path.join("results", "logs", f"{model_name.replace(' ', '_').lower()}_metrics.txt"))
    evaluate_model(model_path, model_path.replace(".pkl", "_scaler.pkl"), X, y, log_path, cv=cv_folds) 


