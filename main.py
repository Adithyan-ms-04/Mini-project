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
from fastapi.staticfiles import StaticFiles
import sys
import io

# Add project root to path for model imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))
from ocular_model import BilateralCKDModel, GrahamTransform
from ensemble_manager import CKDWeightedEnsemble
from input_guardrail import InputGuardrail
from utils import GradCAMPlusPlus
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for models
OCULAR_MODELS = []
CLINICAL_MODEL = None
FEATURE_NAMES = []

def load_all_models():
    global OCULAR_MODELS, CLINICAL_MODEL, FEATURE_NAMES
    
    # 1. Load Ocular Ensemble — All 5 folds (HuggingFace Spaces provides enough RAM)
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
    # Mount frontend AFTER all routes are registered so /predict isn't intercepted
    frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
    if os.path.isdir(frontend_dist):
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
        print(f"Serving frontend from {frontend_dist}")
    else:
        print(f"Warning: frontend/dist not found at {frontend_dist} — API-only mode")

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
        
        # --- WEIGHTED ENSEMBLE (All 5 Folds) ---
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

    # 5. Fusion Evaluation
    # Default is 0.7 Clinical + 0.3 Ocular. You can tune these weights based on model validation!
    CLINICAL_WEIGHT = 0.7
    OCULAR_WEIGHT = 0.3
    final_score = (CLINICAL_WEIGHT * clinical_prob) + (OCULAR_WEIGHT * avg_ocular_prob)

    # 6. Determine Final Risk Tier and Clinical Recommendation
    if final_score < 0.45:
        risk_tier = "Low Risk"
        recommendation = "Routine monitoring recommended. Maintain a balanced diet, stay hydrated, and ensure regular exercise to support kidney health. Continue annual check-ups."
    elif 0.45 <= final_score <= 0.60:
        risk_tier = "Consultation Required (Gray Zone)"
        recommendation = "Inconclusive results. Schedule a follow-up with a nephrologist or primary care physician for a comprehensive metabolic panel (CMP) and urinalysis."
    else:
        risk_tier = "High Risk"
        recommendation = "Immediate clinical follow-up required. Prioritize diagnostic testing including eGFR, BUN, and serum creatinine. Strictly manage blood pressure and blood sugar."
        
    # Inject Lifestyle Advisor AI Recommendations Based on User Input
    lifestyle_advice = []
    clinical_dict = json.loads(clinical_data) if isinstance(clinical_data, str) else clinical_data
    
    if float(clinical_dict.get('Smoking', 0)) == 1:
        lifestyle_advice.append("Quitting smoking is critical to slow CKD progression and reduce cardiovascular risk.")
    if float(clinical_dict.get('AlcoholConsumption', 0)) >= 10:
        lifestyle_advice.append("Reduce alcohol consumption to avoid exacerbating hypertension and kidney stress.")
    if float(clinical_dict.get('PhysicalActivity', 10)) <= 3:
        lifestyle_advice.append("Incorporate at least 150 minutes of moderate aerobic activity weekly to improve metabolic health.")
    if float(clinical_dict.get('DietQuality', 10)) <= 4:
        lifestyle_advice.append("Adopt a kidney-friendly diet low in sodium, processed foods, and refined sugars.")
    if float(clinical_dict.get('SleepQuality', 8)) <= 4:
        lifestyle_advice.append("Improve sleep hygiene; aim for 7-8 hours per night to help regulate blood pressure and stress hormones.")

    if lifestyle_advice:
        recommendation += " Lifestyle Recommendations: " + " ".join(lifestyle_advice)

    # --- LIFESTYLE SIMULATOR ENGINE ---
    # Calculate potential risk reductions if user optimizes their lifestyle.
    lifestyle_simulations = []
    if CLINICAL_MODEL is not None:
        base_df = clinical_df.copy()
        
        # Helper to compute risk drop
        def get_risk_reduction(modified_df):
            sim_clinical_prob = CLINICAL_MODEL.predict_proba(modified_df)[:, 1][0]
            sim_score = (CLINICAL_WEIGHT * sim_clinical_prob) + (OCULAR_WEIGHT * avg_ocular_prob)
            return max(0, final_score - sim_score)
            
        optimal_lifestyle_set = False
        sim_all_df = base_df.copy()

        # 1. Smoking (Optimal: 0)
        if float(base_df['Smoking'].iloc[0]) == 1:
            sim_df = base_df.copy()
            sim_df['Smoking'] = 0.0
            reduction = get_risk_reduction(sim_df)
            if reduction > 0.001:
                lifestyle_simulations.append({
                    "factor": "Smoking",
                    "action": "Quit smoking",
                    "reduction_pct": float(reduction * 100)
                })
            sim_all_df['Smoking'] = 0.0
            optimal_lifestyle_set = True

        # 2. Alcohol (Optimal: 0)
        curr_alcohol = float(base_df['AlcoholConsumption'].iloc[0])
        if curr_alcohol > 0:
            sim_df = base_df.copy()
            sim_df['AlcoholConsumption'] = 0.0
            reduction = get_risk_reduction(sim_df)
            if reduction > 0.001:
                lifestyle_simulations.append({
                    "factor": "Alcohol",
                    "action": "Eliminate alcohol consumption",
                    "reduction_pct": float(reduction * 100)
                })
            sim_all_df['AlcoholConsumption'] = 0.0
            optimal_lifestyle_set = True

        # 3. Physical Activity (Optimal: 10)
        curr_activity = float(base_df['PhysicalActivity'].iloc[0])
        if curr_activity < 10:
            sim_df = base_df.copy()
            sim_df['PhysicalActivity'] = 10.0
            reduction = get_risk_reduction(sim_df)
            if reduction > 0.001:
                lifestyle_simulations.append({
                    "factor": "Exercise",
                    "action": "Increase physical activity to optimal levels",
                    "reduction_pct": float(reduction * 100)
                })
            sim_all_df['PhysicalActivity'] = 10.0
            optimal_lifestyle_set = True

        # 4. Diet Quality (Optimal: 10)
        curr_diet = float(base_df['DietQuality'].iloc[0])
        if curr_diet < 10:
            sim_df = base_df.copy()
            sim_df['DietQuality'] = 10.0
            reduction = get_risk_reduction(sim_df)
            if reduction > 0.001:
                lifestyle_simulations.append({
                    "factor": "Diet",
                    "action": "Adopt an excellent, kidney-friendly diet",
                    "reduction_pct": float(reduction * 100)
                })
            sim_all_df['DietQuality'] = 10.0
            optimal_lifestyle_set = True

        # 5. Sleep Quality (Optimal: 10)
        curr_sleep = float(base_df['SleepQuality'].iloc[0])
        if curr_sleep < 10:
            sim_df = base_df.copy()
            sim_df['SleepQuality'] = 10.0
            reduction = get_risk_reduction(sim_df)
            if reduction > 0.001:
                lifestyle_simulations.append({
                    "factor": "Sleep",
                    "action": "Ensure 8+ hours of high-quality sleep nightly",
                    "reduction_pct": float(reduction * 100)
                })
            sim_all_df['SleepQuality'] = 10.0
            optimal_lifestyle_set = True

        # Combined Optimal
        if optimal_lifestyle_set and len(lifestyle_simulations) > 1:
            total_reduction = get_risk_reduction(sim_all_df)
            if total_reduction > 0.001:
                lifestyle_simulations.append({
                    "factor": "All Changes",
                    "action": "Combined optimal lifestyle",
                    "reduction_pct": float(total_reduction * 100),
                    "is_combined": True
                })

    return {
        "final_risk": float(final_score),
        "ocular_risk": float(avg_ocular_prob),
        "clinical_risk": float(clinical_prob),
        "risk_tier": risk_tier,
        "recommendation": recommendation,
        "saliency_map": saliency_base64,
        "lifestyle_simulations": lifestyle_simulations
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
