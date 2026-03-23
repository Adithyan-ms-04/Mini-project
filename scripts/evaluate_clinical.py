import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

def evaluate_clinical_model(csv_path, model_path, output_dir="models"):
    print(f"Loading clinical dataset from {csv_path}...")
    if not os.path.exists(csv_path):
        # Fallback to parent dir if run from scripts/
        csv_path = os.path.join("..", csv_path)
    
    df = pd.read_csv(csv_path)
    
    CORE_FEATURES = [
        'GFR', 'SerumCreatinine', 'HbA1c', 'SystolicBP', 'Age', 
        'BMI', 'HemoglobinLevels', 'ProteinInUrine', 'ACR', 'BUNLevels'
    ]
    
    if not os.path.exists(model_path):
        model_path_alt = os.path.join("..", model_path)
        if os.path.exists(model_path_alt):
            model_path = model_path_alt
            output_dir = os.path.join("..", output_dir)
        else:
            print(f"Model not found at {model_path}. Please train it first.")
            return

    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    # Check if necessary target column exists
    if 'Diagnosis' not in df.columns:
        print("Error: 'Diagnosis' column not found in dataset.")
        return

    X = df[CORE_FEATURES]
    y = df['Diagnosis']
    
    # Recreate the exact 80/20 train/test split used during training (random_state=42)
    print("Running evaluation on the 20% hold-out test set (random_state=42)...")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*40)
    print("      CLINICAL MODEL EVALUATION")
    print("="*40)
    print("Classification Report:")
    print(classification_report(y_test, preds))
    auc_score = roc_auc_score(y_test, probs)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print("="*40)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Clinical Model Confusion Matrix (Test Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = os.path.join(output_dir, 'clinical_cm.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix: {cm_path}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Clinical)')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(output_dir, 'clinical_roc.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve: {roc_path}")

if __name__ == "__main__":
    clinical_csv = "Chronic_Kidney_Dsease_data.csv"
    model_path = os.path.join("models", "clinical_rf_model.joblib")
    evaluate_clinical_model(clinical_csv, model_path)
