import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Input and Output directories (SageMaker Processing paths)
INPUT_DIR = "/opt/ml/processing/input/"
OUTPUT_DIR = "/opt/ml/processing/output/"

# Load dataset from SageMaker Processing input path
def load_data():
    file_path = os.path.join(INPUT_DIR, "creditcard.csv")
    print(f"Loading dataset from: {file_path}")
    return pd.read_csv(file_path)

# Perform data preprocessing
def preprocess_data(df):
    print(f"Initial dataset shape: {df.shape}")
    # Handling missing values
    print(df.isnull().sum())

    # Splitting features & target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Apply SMOTE For Feature & Target
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Normalize 'Time' & 'Amount'
    scaler = StandardScaler()
    X_resampled[['Time', 'Amount']] = scaler.fit_transform(X_resampled[['Time', 'Amount']])

    # Merge back the Feature & Target
    df_resampled = pd.concat([X_resampled, pd.Series(y_resampled, name="Class")], axis=1)

    # Train-Test Split
    train, test = train_test_split(df_resampled, test_size=0.2, stratify=df_resampled['Class'], random_state=42)
    print(f"Processed Train Shape: {train.shape}, Test shape: {test.shape}")

    # Save processed files inside SageMaker Processing output directory
    train_file = os.path.join(OUTPUT_DIR, "train.csv")
    test_file = os.path.join(OUTPUT_DIR, "test.csv")
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    print(f"Train data saved to: {train_file}")
    print(f"Test data saved to: {test_file}")

if __name__ == "__main__":
    df = load_data()
    preprocess_data(df)