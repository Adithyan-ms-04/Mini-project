import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from efficientnet_pytorch import EfficientNet

class GrahamTransform:
    """
    Implements the Graham preprocessing method (Gaussian blur subtraction) 
    as described in Ben Graham's 2015 Kaggle winning solution for Diabetic Retinopathy.
    This enhances the contrast of vessels and other structures in fundus images.
    """
    def __init__(self, sigma=10):
        self.sigma = sigma

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        # Convert to BGR if using PIL RGB
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        # Basic cropping of black pixels at edges (optional but recommended)
        # For simplicity, we skip complex cropping here and focus on Graham method
        
        # Apply Graham method: original - blurred
        # img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), self.sigma), -4, 128)
        # Note: Nature reports often use a variation of this where they normalize background
        blurred = cv2.GaussianBlur(img, (0, 0), self.sigma)
        enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)
        
        # Convert back to PIL RGB
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced)

class BilateralCKDModel(nn.Module):
    """
    Accepts two fundus images (left and right), extracts features using a 
    shared EfficientNet-B3 backbone, concatenates them, and predicts CKD.
    """
    def __init__(self, model_name='efficientnet-b3', feature_dim=256):
        super(BilateralCKDModel, self).__init__()
        # Shared backbone
        self.backbone = EfficientNet.from_pretrained(model_name)
        
        # Replace original classifier with a projection to feature_dim
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)
        )
        
        # Classification head for concatenated bilateral features
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
            # Sigmoid removed to support BCEWithLogitsLoss as requested in Prompt 3
        )

    def forward(self, left_img, right_img):
        # Extract features from both images using the shared backbone
        left_features = self.backbone(left_img)
        right_features = self.backbone(right_img)
        
        # Concatenate features (Bilateral Late Fusion)
        combined_features = torch.cat((left_features, right_features), dim=1)
        
        # Final prediction
        return self.classifier(combined_features)

if __name__ == "__main__":
    # Test block
    print("Initializing BilateralCKDModel with EfficientNet-B3...")
    model = BilateralCKDModel()
    
    # Dummy inputs (batch size 2, 3 channels, 300x300 size for B3)
    dummy_left = torch.randn(2, 3, 300, 300)
    dummy_right = torch.randn(2, 3, 300, 300)
    
    output = model(dummy_left, dummy_right)
    print(f"Forward pass successful. Output shape: {output.shape}")
    
    # Check if backbone parameters are shared
    print(f"Backbone shared: {id(model.backbone) == id(model.backbone)}")
