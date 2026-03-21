# CKD AI: Multimodal Chronic Kidney Disease Diagnostic Platform

**CKD AI** is an advanced, dual-branch diagnostic architecture designed to predict Chronic Kidney Disease risk by performing a late-fusion ensemble approach. The platform evaluates 10 core biological metrics using a Calibrated Random Forest (Clinical Branch) alongside Retinal Fundus Imaging processed via an EfficientNet-B3 architecture utilizing Graham Transforms (Ocular Branch).

### Development Team
*   **Adithyan M S** - Lead AI Architect (Ocular & Clinical Late-Fusion Ensembling)
*   **Alan Seby** - Frontend Developer & UI/UX Designer
*   **Haripriya B** - Backend Optimization & Data Pipeline Engineer
*   **Lakshmipriya S** - Clinical Data & Guardrails Engineer

---

## 📁 Repository Structure

```text
ckd-ai/
├── frontend/                     # React + Vite + Tailwind CSS User Interface
│   ├── src/
│   │   ├── App.tsx               # Main React Dashboard and Prediction Logic
│   │   ├── Docs.tsx              # Team & Architecture Documentation
│   │   └── index.css             # Glassmorphism and Utility Classes
│   ├── package.json              # Node.js dependencies
│   └── vite.config.ts            # Fast refresh & FastAPI proxy configuration
│
├── models/                       # Storage for large pre-trained machine learning weights
│   ├── clinical_rf_model.joblib  # Core Calibrated Random Forest model
│   ├── clinical_rf_model_features.json # Clinical biomarker tracking map
│   └── fold_*.pth                # PyTorch Convolutional Neural Networks (EfficientNet-B3)
│
├── scripts/                      # AI Pipelines & Model Training Source Code
│   ├── clinical_model.py         # Tabular data training and calibration logic 
│   ├── train_ocular_v2.py        # Retinal fundus training pipeline
│   ├── ensemble_manager.py       # Late-Fusion Logic (Weighted Reliability Score)
│   ├── input_guardrail.py        # OOD Validation and biological anomaly detection
│   └── ocular_model.py           # Neural network PyTorch class definitions
│
├── web_app/                      
│   └── main.py                   # FastAPI Backend serving predictions and model inference
│
├── requirements.txt              # Python Backend dependencies
└── README.md                     # You are here!
```

---

## 🚀 Step-by-Step Installation Guide

To run this application locally on a new PC, you need to spin up **both** the AI backend and the React frontend simultaneously. 

### Prerequisites
*   **Python (v3.10+)**: Required to run the PyTorch and Scikit-Learn backend.
*   **Node.js (v18+)**: Required to run `npm` for the React web application.
*   **Git**: Required to clone this repository.

### Step 1: Clone the Repository
Open your terminal and clone the source code to your machine:
```bash
git clone <your-repository-url>
cd <repository-directory>
```

### Step 2: Boot up the AI Backend (FastAPI)
The backend loads the heavy `.pth` neural networks into memory and exposes the `/predict` API endpoint.

1. **Create a virtual environment (Recommended)**:
   ```bash
   python -m venv venv
   ```
2. **Activate the environment**:
   *   *(Windows)*: `venv\Scripts\activate`
   *   *(Mac/Linux)*: `source venv/bin/activate`
3. **Install AI Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Start the Prediction Engine**:
   ```bash
   python web_app/main.py
   ```
   *The backend should now say it is actively listening natively on Port 8000.*

### Step 3: Boot up the Frontend (React Dashboard)
Leave the backend terminal running, and open a **New Terminal Window**. Navigate back into the project.

1. **Enter the frontend folder**:
   ```bash
   cd frontend
   ```
2. **Install Node dependencies**:
   ```bash
   npm install
   ```
3. **Start the Vite Development Server**:
   ```bash
   npm run dev
   ```

### Step 4: Run Your First Prediction
1. Open your browser and navigate to the Local URL provided by Vite (usually `http://localhost:5173`).
2. Explore the beautifully animated UI! Fill out the Clinical Indicators with test numbers (e.g. GFR: 65, Creatinine: 1.2).
3. Upload a sample Retinal Fundus image. Ensure the image is a valid fundus scan—our biological **InputGuardrails** actively block selfies and screenshots!
4. Click **Run Fusion Analysis**.

---

### Understanding the Model Guardrails
To prevent Artificial Intelligence "hallucinations," this system utilizes an integrated `InputGuardrail`:
1. **OOD Detection**: It calculates Standard Deviation across individual decision trees in the Random Forest. If uncertainty is too high, it rejects the computation.
2. **Biological Reality Checks**: Blocks impossible biological figures directly mapping to human physical limits (e.g. `GFR = 500`).
3. **RGB Heuristics**: Lightweight pre-checks ensuring the image hue distribution matches ocular scans.
