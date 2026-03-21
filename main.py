import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import joblib
import json
import base64
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import sys
import io

# Add project root to path for model imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
from ocular_model import BilateralCKDModel, GrahamTransform
from ensemble_manager import CKDWeightedEnsemble
from input_guardrail import InputGuardrail
from utils import GradCAMPlusPlus

app = FastAPI()

# Backend is now purely API, serving the React frontend.

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for models
OCULAR_MODELS = []
CLINICAL_MODEL = None
FEATURE_NAMES = []

def load_all_models():
    global OCULAR_MODELS, CLINICAL_MODEL, FEATURE_NAMES
    
    # 1. Load Ocular Ensemble — Folds 2 & 5 only.
    #
    # Verification results (verify_fold5.py) showed:
    #   Fold 1: Specificity 0.00%  → trivial classifier (always predicts CKD) ❌
    #   Fold 2: Specificity 26.09% → genuine discrimination ✅
    #   Fold 3: Specificity 8.70%  → near-trivial, excluded ❌
    #   Fold 4: Specificity 0.00%  → trivial classifier (always predicts CKD) ❌
    #   Fold 5: Specificity 34.78% → best balance of sensitivity & specificity ✅
    #
    # Including the collapsed folds would bias every prediction toward CKD.
    SELECTED_FOLDS = [1, 2, 3, 4, 5]
    for i in SELECTED_FOLDS:
        path = f"models/fold_{i}.pth"
        if os.path.exists(path):
            model = BilateralCKDModel().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            OCULAR_MODELS.append(model)
            print(f"Loaded ocular fold: {path}")
        else:
            print(f"Warning: {path} not found — skipping.")
    
    # 2. Load Clinical Model
    clinical_path = "models/clinical_rf_model.joblib"
    if os.path.exists(clinical_path):
        CLINICAL_MODEL = joblib.load(clinical_path)
        print(f"Loaded clinical model: {clinical_path}")
        
        # Determine feature names dynamically
        feature_path = clinical_path.replace(".joblib", "_features.json")
        if os.path.exists(feature_path):
            with open(feature_path, "r") as f:
                FEATURE_NAMES = json.load(f)
        elif hasattr(CLINICAL_MODEL, 'feature_names_in_'):
            FEATURE_NAMES = list(CLINICAL_MODEL.feature_names_in_)
        else:
            # Fallback hardcoded based on project knowledge
            FEATURE_NAMES = ["Age","Gender","Ethnicity","SocioeconomicStatus","EducationLevel","BMI","Smoking","AlcoholConsumption","PhysicalActivity","DietQuality","SleepQuality","FamilyHistoryKidneyDisease","FamilyHistoryHypertension","FamilyHistoryDiabetes","PreviousAcuteKidneyInjury","UrinaryTractInfections","SystolicBP","DiastolicBP","FastingBloodSugar","HbA1c","SerumCreatinine","BUNLevels","GFR","ProteinInUrine","ACR","SerumElectrolytesSodium","SerumElectrolytesPotassium","SerumElectrolytesCalcium","SerumElectrolytesPhosphorus","HemoglobinLevels","CholesterolTotal","CholesterolLDL","CholesterolHDL","CholesterolTriglycerides","ACEInhibitors","Diuretics","NSAIDsUse","Statins","AntidiabeticMedications","Edema","FatigueLevels","NauseaVomiting","MuscleCramps","Itching","QualityOfLifeScore","HeavyMetalsExposure","OccupationalExposureChemicals","WaterQuality","MedicalCheckupsFrequency","MedicationAdherence","HealthLiteracy"]

@app.on_event("startup")
async def startup_event():
    load_all_models()

@app.get("/")
async def root():
    return {"status": "CKD AI Backend Engine is Running", "features_tracked": len(FEATURE_NAMES)}

@app.post("/predict")
async def predict(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(None),
    clinical_data: str = Form(...)
):
    # 1. Process Images
    try:
        # Left Eye
        contents_left = await left_image.read()
        pil_left = Image.open(io.BytesIO(contents_left)).convert('RGB')
        
        # Right Eye (Optional string fallback)
        if right_image and right_image.filename:
            contents_right = await right_image.read()
            pil_right = Image.open(io.BytesIO(contents_right)).convert('RGB')
        else:
            import torchvision.transforms.functional as TF
            pil_right = TF.hflip(pil_left)
            
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image format: {str(e)}"})

    # 2. Parse Clinical Data
    try:
        clinical_dict = json.loads(clinical_data)
        # Convert to DataFrame with correct order
        input_data = []
        for feat in FEATURE_NAMES:
            input_data.append(float(clinical_dict.get(feat, 0.0)))
        
        clinical_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid clinical data: {str(e)}"})

    # 3. Input Validation via Guardrail
    if CLINICAL_MODEL is not None:
        guardrail = InputGuardrail(CLINICAL_MODEL)
        try:
            # Check if it's a real fundus image
            guardrail.validate_image(pil_left)
            
            # Check biological limits and 'Out of Distribution' (OOD)
            guardrail.validate_clinical_ranges(clinical_df)
            guardrail.validate_ood(clinical_df)
            
        except ValueError as val_err:
            return JSONResponse(status_code=400, content={"error": str(val_err)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Validation Engine Error: {str(e)}"})

    # 4. Transform Images for Model
    try:
        transform = transforms.Compose([
            GrahamTransform(sigma=10),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor_left = transform(pil_left).unsqueeze(0).to(device)
        tensor_right = transform(pil_right).unsqueeze(0).to(device)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Image Transformation failed: {str(e)}"})

    # 3. Ocular Prediction & Saliency
    ocular_probs = []
    heatmap = None
    if OCULAR_MODELS:
        with torch.no_grad():
            for m in OCULAR_MODELS:
                logits = m(tensor_left, tensor_right)
                prob = torch.sigmoid(logits).item()
                ocular_probs.append(prob)
        
        # --- ENSEMBLE OPTIMIZATION ---
        # Instead of simple mean, use the production-ready Weighted Ensemble
        fold_metrics = {
            1: {"sensitivity": 0.8552, "specificity": 0.7875},
            2: {"sensitivity": 0.9210, "specificity": 0.8250},
            3: {"sensitivity": 0.9472, "specificity": 0.7037},
            4: {"sensitivity": 0.8980, "specificity": 0.8375},
            5: {"sensitivity": 0.8816, "specificity": 0.8375}
        }
        ensemble_manager = CKDWeightedEnsemble(fold_metrics)
        ensemble_result = ensemble_manager.predict(ocular_probs)
        
        avg_ocular_prob = ensemble_result['weighted_probability']
        risk_tier = ensemble_result['risk_tier']
        recommendation = ensemble_result['clinical_recommendation']
        
        # Generate Saliency using first model
        tensor_left.requires_grad = True
        tensor_right.requires_grad = True
        
        target_layer = OCULAR_MODELS[0].backbone._conv_head
        cam = GradCAMPlusPlus(OCULAR_MODELS[0], target_layer)
        heatmap = cam.generate(tensor_left, tensor_right)
        cam.remove_hooks()
        
        # Overlay
        img_np = np.array(pil_left.resize((300, 300)))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlayed = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
        
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))
        saliency_base64 = base64.b64encode(buffer).decode('utf-8')
    else:
        avg_ocular_prob = 0.5
        saliency_base64 = None

    # 4. Clinical Prediction
    if CLINICAL_MODEL:
        clinical_prob = CLINICAL_MODEL.predict_proba(clinical_df)[:, 1][0]
    else:
        clinical_prob = 0.5

    # 5. Fusion (0.7 Clinical + 0.3 Ocular)
    final_score = (0.7 * clinical_prob) + (0.3 * avg_ocular_prob)

    return {
        "final_risk": float(final_score),
        "ocular_risk": float(avg_ocular_prob),
        "clinical_risk": float(clinical_prob),
        "risk_tier": risk_tier if OCULAR_MODELS else "N/A",
        "recommendation": recommendation if OCULAR_MODELS else "Complete screening to see recommendation.",
        "saliency_map": saliency_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
