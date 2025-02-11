import json
import os
import pandas as pd
#load json dataset
json_directory = "../ABC/dataset/data"
#load groundtruth excel
ground_directory = "../ABC/dataset/groundtruth"
ground_path=os.path.join(ground_directory, "phq9.csv")
groundtruth=pd.read_csv(ground_path)
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
csv_dir=os.path.abspath("../ABC/dataset/processed")
csv_path=os.path.join(csv_dir, "features_raw.csv")
df.to_csv(csv_path, index=False)
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
csv_dir=os.path.abspath("../ABC/dataset/processed")
csv_path=os.path.join(csv_dir, "features_processed.csv")
df.to_csv(csv_path, index=False)

#clean ground truth data
dropped_pids = groundtruth.loc[groundtruth["end_phq9"].isna(), "pid"].tolist()
groundtruth=groundtruth.dropna(subset=["end_phq9"]).reset_index(drop=True)
# format dates
groundtruth["start_ts"]=pd.to_datetime(groundtruth["start_ts"], errors="coerce")
groundtruth["end_ts"]=pd.to_datetime(groundtruth["end_ts"], errors="coerce")
#drop one instance of p17 missing from dataset
groundtruth = groundtruth[~((groundtruth["pid"] == 'P17') & (groundtruth["start_phq9"] == 14))].reset_index(drop=True)
#save to csv
csv_dir=os.path.abspath("../ABC/dataset/processed")
csv_path=os.path.join(csv_dir, "groudtruth_processed.csv")
groundtruth.to_csv(csv_path, index=False)

#matching dates between ground truth and data set
# Merge datasets on 'pid' (inner join keeps only matching pids)
merged_df = df_extracted.merge(groundtruth[["pid", "start_ts", "end_ts"]], on="pid", how="inner")
# Filter rows where 'date' is within 'start_ts' and 'end_ts'
filtered_df = merged_df[(merged_df["date"] >= merged_df["start_ts"]) & (merged_df["date"] <= merged_df["end_ts"])]
# Group by pid, start_ts, and end_ts, then compute the mean of all numeric columns
aggregated_df = filtered_df.groupby(["pid", "start_ts", "end_ts"]).mean(numeric_only=True).reset_index()
#save to csv
csv_dir=os.path.abspath("../ABC/dataset/processed")
csv_path=os.path.join(csv_dir, "features_aggregated.csv")
aggregated_df.to_csv(csv_path, index=False)

#merge groundtruth and dataset
merged_df = aggregated_df.merge(groundtruth[["pid", "start_ts", "end_ts", "depression_episode"]], on=["pid", "start_ts", "end_ts"], how="inner")
#save to csv
csv_dir=os.path.abspath("../ABC/dataset/processed")
csv_path=os.path.join(csv_dir, "features_groundtruth_merged.csv")
merged_df.to_csv(csv_path, index=False)
#train and test data
#key feature extraction and target extraction
from sklearn.model_selection import train_test_split
key_features=["classification_smilingProbability", "au_AU01", "au_AU02", "au_AU06"]
X_data=merged_df[key_features]
target=["depression_episode"]
y_data=merged_df[target]
#save to csv
csv_path=os.path.join(csv_dir, "X.csv")
X_data.to_csv(csv_path, index=False)
csv_path=os.path.join(csv_dir, "y.csv")
y_data.to_csv(csv_path, index=False)




