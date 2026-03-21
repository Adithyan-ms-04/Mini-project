import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler

class CKDLateFusionDataset(Dataset):
    def __init__(self, ocular_csv, clinical_csv, img_dir, transform=None):
        self.ocular_df = pd.read_csv(ocular_csv)
        self.clinical_df = pd.read_csv(clinical_csv)
        self.img_dir = img_dir
        self.transform = transform
        
        # Merge datasets on ID
        # Assuming ID in ocular_df matches PatientID in clinical_df
        self.data = pd.merge(
            self.ocular_df, 
            self.clinical_df, 
            left_on='ID', 
            right_on='PatientID', 
            how='inner'
        )
        
        # Define clinical features (exhaustive list for better accuracy)
        self.clinical_feature_cols = [
            'Age', 'Gender', 'Ethnicity', 'SocioeconomicStatus', 'EducationLevel', 
            'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 
            'DietQuality', 'SleepQuality', 'FamilyHistoryKidneyDisease', 
            'FamilyHistoryHypertension', 'FamilyHistoryDiabetes', 
            'PreviousAcuteKidneyInjury', 'UrinaryTractInfections', 
            'SystolicBP', 'DiastolicBP', 'FastingBloodSugar', 'HbA1c', 
            'SerumCreatinine', 'BUNLevels', 'GFR', 'ProteinInUrine', 
            'ACR', 'SerumElectrolytesSodium', 'SerumElectrolytesPotassium', 
            'SerumElectrolytesCalcium', 'SerumElectrolytesPhosphorus', 
            'HemoglobinLevels', 'CholesterolTotal', 'CholesterolLDL', 
            'CholesterolHDL', 'CholesterolTriglycerides', 'ACEInhibitors', 
            'Diuretics', 'NSAIDsUse', 'Statins', 'AntidiabeticMedications', 
            'Edema', 'FatigueLevels', 'NauseaVomiting', 'MuscleCramps', 
            'Itching', 'QualityOfLifeScore', 'HeavyMetalsExposure', 
            'OccupationalExposureChemicals', 'WaterQuality', 
            'MedicalCheckupsFrequency', 'MedicationAdherence', 'HealthLiteracy'
        ]
        
        # Add ocular indicators as extra features if needed, but usually they are 
        # part of the "ocular" data. For late fusion CKD prediction, 
        # labels like DR/HR are extremely relevant.
        self.clinical_feature_cols += ['DR', 'HR', 'BRVO', 'CRVO']
        
        self.X_clinical = self.data[self.clinical_feature_cols].values
        self.y = self.data['Diagnosis'].values
        
        # Scaling
        self.scaler = StandardScaler()
        self.X_clinical = self.scaler.fit_transform(self.X_clinical)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_id = self.data.iloc[idx]['ID']
        img_path = os.path.join(self.img_dir, f"{int(img_id)}.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Clinical data
        clinical_data = torch.tensor(self.X_clinical[idx], dtype=torch.float32)
        
        # Label
        label = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0)
        
        return image, clinical_data, label

def get_dataloaders(ocular_csv, clinical_csv, img_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = CKDLateFusionDataset(ocular_csv, clinical_csv, img_dir, transform=transform)
    
    # Train-test split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, len(dataset.clinical_feature_cols)
