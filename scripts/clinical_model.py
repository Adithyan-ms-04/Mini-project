import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

def create_clinical_pipeline(n_estimators=100, random_state=42):
    """
    Creates a Scikit-Learn pipeline with:
    - SimpleImputer (median) for missing values
    - StandardScaler for normalization
    - Calibrated Random Forest (Smooth probabilities)
    """
    base_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,           # More restrictive depth
        min_samples_leaf=50,   # High regularization: force look at large groups
        random_state=random_state
    )
    
    # Wrap in Calibrator to 'squish' probabilities into realistic ranges
    calibrated_rf = CalibratedClassifierCV(estimator=base_rf, method='sigmoid', cv=5)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', calibrated_rf)
    ])
    return pipeline

def train_and_save_model(csv_path, model_save_path):
    print(f"Loading clinical data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # --- FEATURE OPTIMIZATION: Core Diagnostic Suite ---
    # We prune from 51 features down to the top 10 biomarkers to prevent 
    # 'Feature Dilution' and ensure the model responds sharply to key indicators.
    CORE_FEATURES = [
        'GFR', 'SerumCreatinine', 'HbA1c', 'SystolicBP', 'Age', 
        'BMI', 'HemoglobinLevels', 'ProteinInUrine', 'ACR', 'BUNLevels'
    ]
    
    print(f"Training on pruned Core Feature set: {CORE_FEATURES}")
    
    # Filter for absolutely needed features
    X = df[CORE_FEATURES]
    y = df['Diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Initializing and training Optimized Random Forest Pipeline...")
    pipeline = create_clinical_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Evaluation
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    # Save the pipeline
    print(f"Saving clinical model to {model_save_path}...")
    joblib.dump(pipeline, model_save_path)
    
    import json
    feature_path = model_save_path.replace(".joblib", "_features.json")
    with open(feature_path, "w") as f:
        json.dump(list(X.columns), f)
        
    return pipeline

def predict_ckd_risk(model_path, new_data_df):
    """
    Loads saved model and returns probability scores.
    """
    pipeline = joblib.load(model_path)
    # Ensure columns match training (excluding Diagnosis etc.)
    # In a real scenario, we'd use a shared feature list.
    probs = pipeline.predict_proba(new_data_df)[:, 1]
    return probs

if __name__ == "__main__":
    # Internal test execution
    clinical_csv = "Chronic_Kidney_Dsease_data.csv"
    model_dir = "models"
    model_name = "clinical_rf_model.joblib"
    model_path = os.path.join(model_name) # simplified for scripts/ context
    
    # Adjust paths if run from scripts/
    if not os.path.exists(clinical_csv) and os.path.exists(os.path.join("..", clinical_csv)):
        clinical_csv = os.path.join("..", clinical_csv)
        model_path = os.path.join("..", model_dir, model_name)
    else:
        # Fallback to local
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    train_and_save_model(clinical_csv, model_path)
