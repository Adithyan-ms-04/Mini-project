import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import joblib
from PIL import Image
from torchvision import transforms
import sys

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from ocular_model import BilateralCKDModel, GrahamTransform

# Assuming utils is in the same directory (scripts/)
from utils import GradCAMPlusPlus

def load_models(fold_paths, clinical_model_path, device):
    ocular_folds = []
    for path in fold_paths:
        model = BilateralCKDModel().to(device)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        ocular_folds.append(model)
    
    clinical_model = joblib.load(clinical_model_path) if os.path.exists(clinical_model_path) else None
    return ocular_folds, clinical_model

def get_fusion_prediction(ocular_models, clinical_model, image_path, clinical_data_row, device):
    # Ocular Preprocessing
    transform = transforms.Compose([
        GrahamTransform(sigma=10),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    tensor_img = transform(img).unsqueeze(0).to(device)
    
    # Ocular Ensemble Inference
    ocular_probs = []
    with torch.no_grad():
        import torchvision.transforms.functional as TF
        tensor_right = TF.hflip(tensor_img)
        for model in ocular_models:
            logits = model(tensor_img, tensor_right)
            prob = torch.sigmoid(logits).item()
            ocular_probs.append(prob)
    
    avg_ocular_prob = np.mean(ocular_probs)
    
    # Clinical Inference
    # Assuming clinical_data_row is a DataFrame or dict with correct features
    clinical_prob = clinical_model.predict_proba(clinical_data_row)[:, 1][0] if clinical_model else 0.5
    
    # Late Fusion Evaluation
    # Default is 0.7 Clinical + 0.3 Ocular. Tuning these weights balances feature relevance.
    CLINICAL_WEIGHT = 0.7
    OCULAR_WEIGHT = 0.3
    final_score = (CLINICAL_WEIGHT * clinical_prob) + (OCULAR_WEIGHT * avg_ocular_prob)
    
    return final_score, avg_ocular_prob, clinical_prob

def visualize_saliency(model, image_path, save_path, device):
    transform = transforms.Compose([
        GrahamTransform(sigma=10),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True
    
    # Target last conv layer in EfficientNet-B3
    # In efficientnet-pytorch, it's usually model.backbone._conv_head
    target_layer = model.backbone._conv_head
    cam = GradCAMPlusPlus(model, target_layer)
    
    import torchvision.transforms.functional as TF
    input_tensor_right = TF.hflip(input_tensor)
    heatmap = cam.generate(input_tensor, input_tensor_right)
    
    # Overlay
    img_np = np.array(img.resize((300, 300)))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlayed = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    
    Image.fromarray(overlayed).save(save_path)
    print(f"Saliency map saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fold_paths = [f"models/fold_{i}.pth" for i in range(1, 6)]
    clinical_path = "models/clinical_rf_model.joblib"
    
    # Check if models exist, otherwise use placeholders (for testing script logic)
    print("Loading ensemble models...")
    ocular_models, clinical_model = load_models(fold_paths, clinical_path, device)
    
    # Sample verification run if image exists
    test_img = "dataset/images/1.png"
    if os.path.exists(test_img):
        print(f"Running fusion prediction on {test_img}...")
        # Mock clinical data for verification (replace with real row for actual metrics)
        mock_clinical = pd.DataFrame([np.zeros(51)], columns=clinical_model.feature_names_in_) if clinical_model else None
        
        score, ocular_p, clinical_p = get_fusion_prediction(ocular_models, clinical_model, test_img, mock_clinical, device)
        print(f"Final Combined CKD Risk: {score:.4f}")
        print(f"(Ocular: {ocular_p:.4f}, Clinical: {clinical_p:.4f})")
        
        if len(ocular_models) > 0:
            visualize_saliency(ocular_models[0], test_img, "saliency_map.png", device)
    else:
        print("Sample image not found for test run.")
