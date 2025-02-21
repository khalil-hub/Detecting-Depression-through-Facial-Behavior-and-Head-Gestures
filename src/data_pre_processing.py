import json
import os
import pandas as pd
import numpy as np
#functions  for data loading, processing, feature selection..
def data_extraction(json_directory, data_path):
    # List all JSON files in the directory
    json_files = [f for f in os.listdir(json_directory) if f.endswith('.json')]
    # Create an empty list to store extracted data
    data_list = []
    # Loop through each JSON file
    for json_file in json_files:
        file_path = os.path.join(json_directory, json_file)
        with open(file_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                data_list.extend(data)  # Ensure list of dictionaries
            else:
                data_list.append(data)  # If it's a single dictionary, append it
    # Convert list of JSON objects into a structured DataFrame
    df = pd.json_normalize(data_list, sep="_")
    df.to_csv(data_path, index=False)

def data_pre_processing(df, csv_path):
    columns_to_keep = [
        "pid", "boundingBox", "classification_leftEyeOpenProbability", 
        "classification_rightEyeOpenProbability", "classification_smilingProbability",
        "headEulerAngle_X", "headEulerAngle_Y", "headEulerAngle_Z", "au_AU01", "au_AU02",
        "au_AU04", "au_AU06", "au_AU07", "au_AU10", "au_AU12", "au_AU14", "au_AU15",
        "au_AU17", "au_AU23", "au_AU24", "fileName"
    ]
    # Ensure columns exist before extraction
    df_extracted = df.reindex(columns=columns_to_keep, fill_value=None)
    df_extracted=df_extracted.rename(columns={"fileName": "date"})
    # date formatting
    df_extracted["date"]=df_extracted["date"].str.extract(r'(2.{9})')
    df_extracted["date"]=pd.to_datetime(df_extracted["date"])
    # pid filtering based on ground data pid dropped are: ['P14', 'P25', 'P27', 'P34']
    pids_todrop=['P14', 'P25', 'P27', 'P34']
    df_extracted=df_extracted[~df_extracted["pid"].isin(pids_todrop)].reset_index(drop=True)
    #save to csv
    df_extracted.to_csv(csv_path, index=False)
    
def groundtruth_processed(groundtruth, grounstruth_path):
    #clean ground truth data
    groundtruth=groundtruth.dropna(subset=["end_phq9"]).reset_index(drop=True)
    # format dates
    groundtruth["start_ts"]=pd.to_datetime(groundtruth["start_ts"], errors="coerce")
    groundtruth["end_ts"]=pd.to_datetime(groundtruth["end_ts"], errors="coerce")
    #drop one instance of p17 missing from dataset
    groundtruth = groundtruth[~((groundtruth["pid"] == 'P17') & (groundtruth["start_phq9"] == 14))].reset_index(drop=True)
    #save to csv
    groundtruth.to_csv(grounstruth_path, index=False)

def data_merge_mean(groundtruth, df, merged_data_path):
    #matching dates between ground truth and our data set
    # Merge datasets on 'pid' (inner join keeps only matching pids)
    merged_df = df.merge(groundtruth[["pid", "start_ts", "end_ts"]], on="pid", how="inner")
    # Filter rows where 'date' is within 'start_ts' and 'end_ts'
    filtered_df = merged_df[(merged_df["date"] >= merged_df["start_ts"]) & (merged_df["date"] <= merged_df["end_ts"])]
    # Group by pid, start_ts, and end_ts, then compute the mean of all numeric columns
    aggregated_df = filtered_df.groupby(["pid", "start_ts", "end_ts"]).mean(numeric_only=True).reset_index()
    #include depression episode and ensure matching rows
    merged_df = aggregated_df.merge(groundtruth[["pid", "start_ts", "end_ts", "depression_episode"]], on=["pid", "start_ts", "end_ts"], how="inner")
    #save to csv
    merged_df.to_csv(merged_data_path, index=False)
    
def feature_target(merged_data, X_path, y_path):
    #key feature extraction and target extraction
    key_features=["classification_smilingProbability", "au_AU01", "au_AU02", "au_AU06"]
    X_data=merged_data[key_features]
    target=["depression_episode"]
    y_data=merged_data[target]
    #save to csv
    X_data.to_csv(X_path, index=False)
    y_data.to_csv(y_path, index=False)

def data_merge_timeseries(groundtruth, df, timeseries_data_path):
    #Merge groundtruth and extracted data while keeping the full time-series for each participant
    merged_df = df.merge(groundtruth[["pid", "start_ts", "end_ts", "depression_episode"]], on="pid", how="inner")
    # Keep only rows where 'date' is within 'start_ts' and 'end_ts'
    filtered_df = merged_df[(merged_df["date"] >= merged_df["start_ts"]) & (merged_df["date"] <= merged_df["end_ts"])]
    # Keep full time-series instead of averaging
    filtered_df = filtered_df.sort_values(by=["pid", "date"])
    # Save as CSV
    filtered_df.to_csv(timeseries_data_path, index=False)

def create_sequences(data, sequence_path_X, sequence_path_y, TIME_STEPS):
    #Convert time-series data into sequences for deep learning models
    sequences, labels = [], []
    for pid in data["pid"].unique():
        participant_data = data[data["pid"] == pid].sort_values(by="date")
        for i in range(len(participant_data) - TIME_STEPS):
            seq = participant_data.iloc[i:i+TIME_STEPS, 2:-1].values  # Select feature columns
            label = participant_data.iloc[i+TIME_STEPS, -1]  # Target column
            sequences.append(seq)
            labels.append(label)
    # Convert to NumPy array and save
    X = np.array(sequences)
    y = np.array(labels)
    np.save(sequence_path_X, X)
    np.save(sequence_path_y, y)

def feature_target_timeseries(merged_data, X_sequence_path, y_sequence_path, TIME_STEPS):
    #Extract time-series features and save in sequence format for DL models
    key_features = ["classification_smilingProbability", "au_AU01", "au_AU02", "au_AU06"]
    target = ["depression_episode"]
    # Select relevant columns
    merged_data = merged_data[["pid", "date"] + key_features + target]
    # Convert to sequences
    create_sequences(merged_data, X_sequence_path, y_sequence_path, TIME_STEPS)
