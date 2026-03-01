import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_best_model():
    print("--- Starting Advanced Model Training ---")
    
    # 1. Load the cleaned data
    df = pd.read_csv('data/processed/cleaned_loan_data.csv')

    # 2. Encoding Categorical Variables (User Request)
    # This turns City, Education, etc., into numeric columns
    df = pd.get_dummies(df, drop_first=True)

    # 3. Split Features and Target
    X = df.drop("LoanApproved", axis=1)
    y = df['LoanApproved']
    
    # Save the feature names (CRITICAL for the app to work later)
    model_columns = list(X.columns)
    
    # Stratify=y ensures the 0/1 ratio is same in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Define Models (User Request)
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, class_weight="balanced"),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    }

    best_model = None
    best_score = -np.inf
    best_model_name = None

    # 5. Model Comparison Loop
    print("\nModel Evaluation Results:")
    print("-" * 40)
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f'{model_name}:')
        print(f'  Accuracy: {accuracy:.3f}')
        print(f'  F1 Score: {f1:.3f}')
        print(f'  ROC-AUC:  {roc_auc:.3f}')
        print('-' * 40)
        
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model
            best_model_name = model_name

    print(f"✅ Best Model Selected: {best_model_name} (ROC-AUC: {best_score:.3f})")

    # 6. Save the Best Model and Column List
    os.makedirs('models', exist_ok=True)
    with open('models/best_loan_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/model_columns.pkl', 'wb') as f:
        pickle.dump(model_columns, f)
    
    print("Files saved to /models folder.")

if __name__ == "__main__":
    train_best_model()