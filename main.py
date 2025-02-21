import os
import pandas as pd
from src.data_pre_processing import (data_extraction, data_pre_processing, groundtruth_processed, data_merge_mean,
feature_target, data_merge_timeseries, create_sequences, feature_target_timeseries)
from src.evaluation import evaluate_ml_model
from src.train import ml_training
from sklearn.model_selection import cross_val_score, StratifiedKFold

#define the time step globally 
TIME_STEPS=10

directories={
"json_directory": "../ABC/dataset/data",
"groundtruth_directory": "../ABC/dataset/groundtruth",
"data_processed_directory": "../ABC/dataset/processed",
"sequences": "../ABC/dataset/sequences"
}

paths={
"features_raw": os.path.join(directories["data_processed_directory"], "features_raw.csv"),
"features_pre_processed": os.path.join(directories["data_processed_directory"], "features_pre_processed.csv"),
"groundtruth_path": os.path.join(directories["groundtruth_directory"], "phq9.csv"),
"groundtruth_processed": os.path.join(directories["data_processed_directory"], "groudtruth_processed.csv"),
"features_groundtruth_merged": os.path.join(directories["data_processed_directory"], "features_groundtruth_merged.csv"),
"features_X": os.path.join(directories["data_processed_directory"], "X.csv"),
"target_y": os.path.join(directories["data_processed_directory"], "y.csv"),
"timeseries": os.path.join(directories["sequences"], "timeseries.csv"),
"sequence_X": os.path.join(directories["sequences"], "sequence_X.npy"),
"sequence_y": os.path.join(directories["sequences"], "sequence_y.npy"),
"features_sequence_X": os.path.join(directories["sequences"], "features_sequence_X.npy"),
"target_sequence_y": os.path.join(directories["sequences"], "target_sequence_y.npy")
}

model_paths = {
        "logistic regression": os.path.abspath(os.path.join("../ABC", "models/machine_learning", "logistic_regression.pkl")),
        "random forest": os.path.abspath(os.path.join("../ABC", "models/machine_learning", "random_forest.pkl")),
        "XGBoost": os.path.abspath(os.path.join("../ABC", "models/machine_learning", "XGBoost.pkl")),
        "SVM": os.path.abspath(os.path.join("../ABC", "models/machine_learning", "SVM.pkl"))
    }

def main():
    #check if all files exist
    if all(os.path.exists(paths[f]) for f in paths):
        print("all output files exist. skipping function execution")
        return
    #extract data from json files
    if not os.path.exists(paths["features_raw"]):
        data_extraction(directories["json_directory"], paths["features_raw"])
    #pre process the data
    if not os.path.exists(paths["features_pre_processed"]):
        features_raw=pd.read_csv(paths["features_raw"])
        data_pre_processing(features_raw, paths["features_pre_processed"])
    #process ground truth
    if not os.path.exists(paths["groundtruth_processed"]):
        groundtruth=pd.read_csv(paths["groundtruth_path"])
        groundtruth_processed(groundtruth, paths["groundtruth_processed"])
    #merge data with groundtruth using mean value
    if not os.path.exists(paths["features_groundtruth_merged"]):
        groundtruth_process=pd.read_csv(paths["groundtruth_processed"])
        data_preprocessed=pd.read_csv(paths["features_pre_processed"])
        data_merge_mean(groundtruth_process, data_preprocessed, paths["features_groundtruth_merged"])
    #save X and y excel files
    if not os.path.exists(paths["features_X"]) or not os.path.exists(paths["target_y"]):
        merged_data=pd.read_csv(paths["features_groundtruth_merged"])
        feature_target(merged_data, paths["features_X"], paths["target_y"])
    #time series data merge 
    if not os.path.exists(paths["timeseries"]):
        groundtruth_process=pd.read_csv(paths["groundtruth_processed"])
        data_preprocessed=pd.read_csv(paths["features_pre_processed"])
        data_merge_timeseries(groundtruth_process, data_preprocessed, paths["timeseries"])
    #convert data to sequences of 10 time steps per sequence
    if not os.path.exists(paths["sequence_X"]) or not os.path.exists(paths["sequence_y"]):
        timeseries=pd.read_csv(paths["timeseries"])
        create_sequences(timeseries, paths["sequence_X"], paths["sequence_y"], TIME_STEPS)
    #feature and target time series
    if not os.path.exists(paths["features_sequence_X"]) or not os.path.exists(paths["target_sequence_y"]):
        timeseries=pd.read_csv(paths["timeseries"])
        feature_target_timeseries(timeseries, paths["features_sequence_X"], paths["target_sequence_y"], TIME_STEPS)    

    print("data pipeline execution complete ✅")

if __name__=="__main__":
    #main execution
    main()
    X=pd.read_csv(paths["features_X"])
    target_y=pd.read_csv(paths["target_y"])
    #flatten the y array
    y = target_y.values.ravel()  # Convert to 1D array

    # Cross-validation folds
    cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Iterate over models, train, save, and evaluate
    for model_name, model_path in model_paths.items():
        # Train the model with cross-validation
        model, scaler, mean_auc = ml_training(X, y, model_name, model_path, cv=cv_folds)

        # Evaluate model using cross-validation and log the metrics
        log_path = os.path.abspath(os.path.join("results", "logs", f"{model_name.replace(' ', '_').lower()}_metrics.txt"))
        evaluate_ml_model(model_path, model_path.replace(".pkl", "_scaler.pkl"), X, y, log_path, cv=cv_folds) 

    
