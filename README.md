# HeartDisease
This project explores binary classification on tabular medical data. The model analyzes various health metrics (such as blood pressure, cholesterol, and age) to assess a patient's risk profile. The pipeline focuses heavily on proper data hygiene including feature scaling and stratified data partitioning to prevent data leakage and ensure reliable evaluation.

Tech Stack & Libraries
Python
Pandas & NumPy 
Scikit-Learn 
Matplotlib 

Architecture
Data Preprocessing & Encoding: Maps categorical text labels (Presence/Absence) to binary integers (1/0).
Scaling: Utilizes StandardScaler to normalize all clinical inputs to a standard normal distribution, preventing features with larger numerical ranges from disproportionately skewing the model.
Splitting: Splits the dataset into a 70% Training, 15% Validation, and 15% Test architecture. Uses stratification (stratify=y) to maintain an identical ratio of positive and negative cases across all splits.
Classification: Trains a optimized LogisticRegression model to output clean probabilistic binary predictions.
