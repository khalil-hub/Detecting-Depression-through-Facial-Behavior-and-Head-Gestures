import os
import numpy as np
import pandas as pd
from src.data_pre_processing import (
    data_extraction, data_pre_processing, groundtruth_processed,
    data_merge_mean, feature_target, data_merge_timeseries, create_sequences
)
from src.train import ml_training_lopo, train_lstm_lopo
from src.evaluation import evaluate_ml_model_lopo, evaluate_lstm_lopo

# 🔹 Define global constants
TIME_STEPS = 10

# 🔹 Define directories
directories = {
    "json_directory": "../ABC/dataset/data",
    "groundtruth_directory": "../ABC/dataset/groundtruth",
    "data_processed_directory": "../ABC/dataset/processed",
    "sequences": "../ABC/dataset/sequences",
    "logs": "../ABC/results/logs",
    "models": "../ABC/models"
}

# 🔹 Define file paths
paths = {
    "features_raw": os.path.join(directories["data_processed_directory"], "features_raw.csv"),
    "features_pre_processed": os.path.join(directories["data_processed_directory"], "features_pre_processed.csv"),
    "groundtruth_path": os.path.join(directories["groundtruth_directory"], "phq9.csv"),
    "groundtruth_processed": os.path.join(directories["data_processed_directory"], "groundtruth_processed.csv"),
    "features_groundtruth_merged": os.path.join(directories["data_processed_directory"], "features_groundtruth_merged.csv"),
    "features_X": os.path.join(directories["data_processed_directory"], "X.csv"),
    "target_y": os.path.join(directories["data_processed_directory"], "y.csv"),
    "timeseries": os.path.join(directories["sequences"], "timeseries.csv"),
    "sequence_X": os.path.join(directories["sequences"], "sequence_X.npy"),
    "sequence_y": os.path.join(directories["sequences"], "sequence_y.npy")
}

# 🔹 Define model paths
model_paths = {
    "logistic regression": os.path.join(directories["models"], "logistic_regression.pkl"),
    "random forest": os.path.join(directories["models"], "random_forest.pkl"),
    "XGBoost": os.path.join(directories["models"], "xgboost.pkl"),
    "SVM": os.path.join(directories["models"], "svm.pkl")
}

lstm_model_path = os.path.join(directories["models"], "lstm_model.h5")

# 🔹 Function to check if all files exist
def check_files():
    return all(os.path.exists(paths[f]) for f in paths)

# 🔹 Data Pipeline Execution
def main():
    if check_files():
        print("✅ All output files exist. Skipping function execution.")
        return

    # Step 1: Extract data from JSON files
    if not os.path.exists(paths["features_raw"]):
        data_extraction(directories["json_directory"], paths["features_raw"])

    # Step 2: Pre-process the extracted features
    if not os.path.exists(paths["features_pre_processed"]):
        df = pd.read_csv(paths["features_raw"])
        data_pre_processing(df, paths["features_pre_processed"])

    # Step 3: Process the ground truth
    if not os.path.exists(paths["groundtruth_processed"]):
        groundtruth = pd.read_csv(paths["groundtruth_path"])
        groundtruth_processed(groundtruth, paths["groundtruth_processed"])

    # Step 4: Merge data with ground truth (Mean Aggregation)
    if not os.path.exists(paths["features_groundtruth_merged"]):
        groundtruth = pd.read_csv(paths["groundtruth_processed"])
        data_processed = pd.read_csv(paths["features_pre_processed"])
        data_merge_mean(groundtruth, data_processed, paths["features_groundtruth_merged"])

    # Step 5: Extract feature-target pairs
    if not os.path.exists(paths["features_X"]) or not os.path.exists(paths["target_y"]):
        merged_data = pd.read_csv(paths["features_groundtruth_merged"])
        feature_target(merged_data, paths["features_X"], paths["target_y"])

    # Step 6: Merge time-series data
    if not os.path.exists(paths["timeseries"]):
        groundtruth = pd.read_csv(paths["groundtruth_processed"])
        data_processed = pd.read_csv(paths["features_pre_processed"])
        data_merge_timeseries(groundtruth, data_processed, paths["timeseries"])

    # Step 7: Create sequences for deep learning models
    if not os.path.exists(paths["sequence_X"]) or not os.path.exists(paths["sequence_y"]):
        timeseries = pd.read_csv(paths["timeseries"])
        create_sequences(timeseries, paths["sequence_X"], paths["sequence_y"], TIME_STEPS)

    print("✅ Data pipeline execution complete.")

if __name__ == "__main__":
    main()

    print("\n🚀 Starting LOPO-CV Training and Evaluation for ML Models...\n")

    # Load dataset for ML models
    X = pd.read_csv(paths["features_X"]).values
    y = pd.read_csv(paths["target_y"]).values.ravel()
    pids = pd.read_csv(paths["features_groundtruth_merged"])["pid"].values

    for model_name, model_path in model_paths.items():
        print(f"\n🔄 Training {model_name} with LOPO-CV...")
        
        # Train ML model with LOPO-CV
        model, scaler, mean_auc, std_auc = ml_training_lopo(X, y, pids, model_name, model_path)

        # Define log path
        log_path = os.path.join(directories["logs"], f"{model_name.replace(' ', '_').lower()}_metrics.txt")

        # Evaluate ML model
        evaluate_ml_model_lopo(model_path, model_path.replace(".pkl", "_scaler.pkl"), X, y, pids, log_path)

    print("\n✅ LOPO-CV Training & Evaluation for ML Models Completed!")

    print("\n🚀 Starting LOPO-CV Training and Evaluation for LSTM Model...\n")

    # Load dataset for LSTM
    X_lstm = np.load(paths["sequence_X"])
    y_lstm = np.load(paths["sequence_y"])
    pids_lstm = pd.read_csv(paths["features_groundtruth_merged"])["pid"].values

    # Train LSTM model
    model, mean_auc, std_auc = train_lstm_lopo(X_lstm, y_lstm, pids_lstm, lstm_model_path)

    # Define log path for LSTM
    lstm_log_path = os.path.join(directories["logs"], "lstm_metrics.txt")

    # Evaluate LSTM model
mean_auc, std_auc, mean_acc, std_acc, mean_precision, std_precision = evaluate_lstm_lopo(
    lstm_model_path, lstm_model_path.replace(".h5", "_scaler.pkl"), X_lstm, y_lstm, pids_lstm, lstm_log_path
)


print("\n✅ LOPO-CV Training & Evaluation for LSTM Model Completed!")
