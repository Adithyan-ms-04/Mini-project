import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from ocular_model import BilateralCKDModel, GrahamTransform
from train_ocular_v2 import BilateralDataset
from torchvision import transforms

def evaluate_all_folds(data_csv, clinical_csv, img_dir, model_dir, device):
    # Match the new Ocular-only preprocessing logic
    df = pd.read_csv("dataset/RFMiD_Training_Labels.csv")
    
    # Use the explicit Disease_Risk column as the target
    df['Diagnosis'] = df['Disease_Risk']
    
    skf = StratifiedGroupKFold(n_splits=5)
    groups = df['ID']
    X, y = df.index, df['Diagnosis']
    
    val_transform = transforms.Compose([
        GrahamTransform(sigma=10),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y, groups=groups)):
        fold_num = fold_idx + 1
        model_path = os.path.join(model_dir, f"fold_{fold_num}.pth")
        
        if not os.path.exists(model_path):
            print(f"Skipping Fold {fold_num}: {model_path} not found.")
            continue
            
        print(f"\n--- Evaluating Fold {fold_num} ---")
        
        # Load weights
        model = BilateralCKDModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Prepare Validation Loader
        val_df = df.iloc[val_idx]
        val_ds = BilateralDataset(val_df, img_dir, transform=val_transform)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for left, right, labels in val_loader:
                left, right = left.to(device), right.to(device)
                outputs = model(left, right)
                preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate Metrics
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        print(f"Accuracy: {accuracy:.4f} | Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f}")
        
        # Save results for summary
        results.append({
            'Fold': fold_num,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        })
        
        # Plot Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix - Fold {fold_num}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        save_plot_path = os.path.join(model_dir, f'fold_{fold_num}_cm.png')
        plt.savefig(save_plot_path)
        plt.close()
        print(f"Saved confusion matrix: {save_plot_path}")

    # Final Summary
    print("\n" + "="*40)
    print("      ENSEMBLE EVALUATION SUMMARY")
    print("="*40)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    print("-" * 40)
    print(f"Mean Accuracy:    {summary_df['Accuracy'].mean():.4f}")
    print(f"Mean Sensitivity: {summary_df['Sensitivity'].mean():.4f}")
    print(f"Mean Specificity: {summary_df['Specificity'].mean():.4f}")
    print("="*40)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (adjusting if run from project root)
    data_csv = "data/CKD_Ocular_Training_Set.csv"
    clinical_csv = "Chronic_Kidney_Dsease_data.csv"
    img_dir = "dataset/images"
    model_dir = "models"
    
    evaluate_all_folds(data_csv, clinical_csv, img_dir, model_dir, device)
