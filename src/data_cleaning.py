import pandas as pd
import os

def run_cleaning(input_path, output_path):
    print(f"--- Starting Data Cleaning: {input_path} ---")
    
    # Load data
    df = pd.read_csv(input_path)

    # 1. Handling Missing Values (User Requested)
    # Using assignment instead of inplace=True is recommended in modern Pandas
    df['Income'] = df['Income'].fillna(df['Income'].median())
    df['CreditScore'] = df['CreditScore'].fillna(df['CreditScore'].median())
    df['Education'] = df['Education'].fillna('Unknown')

    # 2. Correcting Logical Errors with Clipping (User Requested)
    # This turns negative values into 0 instead of deleting the row
    df['Income'] = df['Income'].clip(lower=0)
    df['LoanAmount'] = df['LoanAmount'].clip(lower=0)

    # 3. Feature Engineering (MDS Best Practice)
    df['DTI_Ratio'] = df['LoanAmount'] / (df['Income'] + 1)

    # 4. Data Integrity Checks (User Requested)
    print("\n[Check] Education Value Counts:")
    print(df['Education'].value_counts())
    
    print(f"\n[Check] Missing values remaining: {df.isnull().sum().sum()}")

    # 5. Save to Tidy Folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSuccess! Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    # Define paths according to our tidy structure
    RAW_DATA = 'data/raw/loan_risk_prediction_dataset.csv'
    PROCESSED_DATA = 'data/processed/cleaned_loan_data.csv'
    
    run_cleaning(RAW_DATA, PROCESSED_DATA)