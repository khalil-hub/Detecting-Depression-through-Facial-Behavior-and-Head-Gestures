# Detecting-Depression-through-Facial-Behavior-and-Head-Gestures
# 0. Overview
The data set used for this challenge was collected as part of the FacePsy study containing facial behavior and head gesture data to detect depressive episodes.
- FacePsy runs on user's device in the background once installed.
- Triggers data collection when users unlock phones or use apps. 
- Data is collected only for first 10 seconds when the trigger initiated.
- Features Collected: 12 AU, 1 Smile, 2 Eye open, 3 Pose, and 133  landmark points.
# 1. Problem definition
The challenge is to detect **depressive episode** with **facial behavior** and **head gesture** 
A participant observation period (i.e., 2 weeks) is labeled as having a depressive episode(label=1) if and only if the participant reported PHQ-9 >= 5 both at the start and end of the observation period; otherwise, it is labeled a non-depressive (label=0).

# 2. Understanding, cleaning and pre-processing the data
- The data contains participant data in JSON format, capturing facial features and behavior during the study. Each entry represents unique facial expressions and orientations captured during specific user events.
- Features:
  * Action Units (AU): facial muscle contraction intensity
  * BoundingBox: coordinates of face
  * Classification: open/closed probability of eye, mouth
  * Contours: Points outlining detailed facial features
  * File Information: data storage file
  * Head Euler Angle: 3 dimensional head orientation (X, Y, Z)
  * Landmarks: key points of face
  * Metadata: additional details
  * PID: participant ID
  * groundtruth: depressive episode labeling
- Load json files and convert them to structured dataframe
- handle missing data (interpolation, mean..)
- normalize features to ensure consistency for ML model training
- convert categorical data to numerical values

# 3. Exploratory data analysis
* Attributing a score for the app being used by the user (eg. twitter, flower game) and cross correlate with the time of the day being used
* Evaluate feature importance for AU's using random forest
* SVM model training
* phone unlock frequency 

# 4. feature engineering
