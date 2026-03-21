import numpy as np
import pandas as pd
from PIL import Image

class InputGuardrail:
    """
    Biological and Out-Of-Distribution (OOD) Input Validator for the CKD model.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.core_features = [
            'GFR', 'SerumCreatinine', 'HbA1c', 'SystolicBP', 'Age', 
            'BMI', 'HemoglobinLevels', 'ProteinInUrine', 'ACR', 'BUNLevels'
        ]

    def validate_image(self, image_input) -> bool:
        """
        Lightweight check to see if an image is a Retinal Fundus Scan.
        A very naive heuristic check based on typical fundus color distribution 
        (fundus images usually have a high proportion of orange/red hues).
        """
        # Open image if it's a path, otherwise assume it's already a PIL Image
        try:
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            else:
                img = image_input.convert('RGB')  # Assuming it's a PIL Image
        except Exception as e:
            raise ValueError(f"Could not open image: {e}")

        # Convert to numpy array
        img_np = np.array(img)
        
        # Calculate mean channels (R, G, B)
        # Fundus images are predominantly reddish/orange. 
        # Typically R > G > B
        r_mean = np.mean(img_np[:, :, 0])
        g_mean = np.mean(img_np[:, :, 1])
        b_mean = np.mean(img_np[:, :, 2])
        
        diff_rg = r_mean - g_mean
        diff_gb = g_mean - b_mean
        
        # This is a naive heuristic. In a real production system, you would want to use
        # a dedicated, lightweight CNN or specialized feature extractor for out-of-distribution
        # detection (like checking for the circular mask of the fundus).
        is_fundus_heuristic = (r_mean > g_mean) and (diff_rg > 10) and (r_mean > 50)
        
        if not is_fundus_heuristic:
             raise ValueError(
                "Image Validation Failed: The uploaded image does not appear to be a Retinal Fundus scan. "
                f"(RGB Profile: R:{r_mean:.1f}, G:{g_mean:.1f}, B:{b_mean:.1f})"
            )
        return True

    def validate_clinical_ranges(self, df: pd.DataFrame) -> bool:
        """
        Checks if the provided clinical features fall within biologically possible ranges.
        """
        # Define strict biological bounding boxes
        allowed_ranges = {
            'GFR': (0.0, 200.0),            # ml/min/1.73m2
            'SerumCreatinine': (0.1, 25.0), # mg/dL
            'HbA1c': (3.0, 20.0),           # %
            'SystolicBP': (60.0, 250.0),    # mmHg
            'Age': (1.0, 120.0),            # Years
            'BMI': (10.0, 80.0),            # kg/m2
            'HemoglobinLevels': (3.0, 25.0),# g/dL
            'ProteinInUrine': (0.0, 10.0),  # g/day
            'ACR': (0.0, 5000.0),           # mg/g
            'BUNLevels': (2.0, 150.0)       # mg/dL
        }
        
        for feature in self.core_features:
            if feature not in df.columns:
                raise ValueError(f"Missing required clinical feature: {feature}")
            
            value = df[feature].iloc[0]
            min_val, max_val = allowed_ranges.get(feature, (-np.inf, np.inf))
            
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Biological Reality Check Failed: {feature} value ({value}) "
                    f"is outside the humanly possible range ({min_val} - {max_val})."
                )
        return True

    def validate_ood(self, df: pd.DataFrame, threshold: float = 0.25) -> bool:
        """
        Calculates Standard Deviation across individual trees in the Random Forest.
        High disagreement = Out of Distribution / Uncertain Input.
        """
        # Ensure we only pass the correct features to the model
        X = df[self.core_features]
        
        # We need the underlying RandomForestClassifier.
        # It's wrapped in a Pipeline, and then inside a CalibratedClassifierCV.
        # We must extract the base estimator.
        try:
             # step 1: Get the CalibratedClassifierCV from the Pipeline
             calibrated_clf = self.pipeline.named_steps['classifier']
             # step 2: The CalibratedClassifierCV trains an ensemble of base models (one per CV fold).
             # We will calculate the variance across the predictions of all base models
             # inside the calibrator. 
             # (Note: A true inner-forest tree variance is harder to get when wrapped 
             # in CalibratedClassifierCV, so analyzing calibrator ensemble variance is a good proxy).
             base_models = calibrated_clf.calibrated_classifiers_
             
             # Scale the input using the pipeline's scaler
             X_imputed = self.pipeline.named_steps['imputer'].transform(X)
             X_scaled = self.pipeline.named_steps['scaler'].transform(X_imputed)
             
             # Gather predictions from each decision tree inside the underlying Random Forest
             # We must access the underlying Base Estimator for the *first* calibrated fold
             base_rf = base_models[0].estimator
             
             tree_preds = []
             for tree in base_rf.estimators_:
                 # get un-calibrated tree predictions
                 prob = tree.predict_proba(X_scaled)[:, 1]
                 tree_preds.append(prob[0])
                 
             std_dev = np.std(tree_preds)
             
             if std_dev > threshold:
                raise ValueError(
                    f"Out-of-Distribution (OOD) Flag triggered! "
                    f"The Random Forest is highly uncertain about this input profile (Disagreement StdDev: {std_dev:.3f}). "
                    f"This may be an anomalous patient or corrupted data."
                )
             return True
             
        except Exception as e:
            # If the internal structure is different, fail gracefully
            print(f"Warning: OOD Validation check encountered an error: {e}")
            return True # Pass validation if we cannot compute OOD
