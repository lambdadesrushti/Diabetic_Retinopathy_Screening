# 👁️ RetinaGuard AI: Diabetic Retinopathy & Multimodal Clinical Risk Screening

> **Research Prototype:** An explainable clinical decision support suite that integrates deep learning fundus image analysis with metabolic risk factors to assist early diabetic retinopathy screening.

---

## 📌 Overview

Diabetic Retinopathy (DR) is a primary cause of preventable visual impairment in working-age populations worldwide. Traditional screening workflows face critical bottlenecks due to specialist shortages and manual grading delays.

**RetinaGuard AI** bridges this gap by providing an end-to-end, multi-tier screening system:
1. **Automated Fundus Analysis:** Grades retinal damage across the 5-point International Clinical Diabetic Retinopathy scale.
2. **Explainable AI (XAI):** Visualizes convolutional feature activations using **Grad-CAM** alongside **U-Net** lesion masks and **ResNet** anatomical landmarks.
3. **Multimodal Metabolic Assessment:** Fuses retinal grading with patient metabolic risk parameters (HbA1c, systolic blood pressure, and diabetes duration) at decision level.
4. **Clinical Decision Support:** Delivers instantaneous risk-stratified follow-up recommendations and exportable clinical PDF reports.

---

## 📸 Application Screenshots
<img width="958" height="499" alt="image" src="https://github.com/user-attachments/assets/e73bb5e9-0cdb-4cff-8104-cb66c69a94cd" />
<img width="958" height="498" alt="image" src="https://github.com/user-attachments/assets/31a5de1a-1b93-4b04-a6f1-d482618abca8" />
<img width="944" height="485" alt="image" src="https://github.com/user-attachments/assets/2b977183-4173-4292-a64f-199d7f14a78e" />

---

## 🧠 System Architecture & Pipeline

```text
 ┌──────────────────┐
 │  Retinal Fundus  │ ──► CLAHE Preprocessing ──► EfficientNet-B0 (Grade 0-4 + Grad-CAM)
 │      Image       │                        ──► U-Net (Lesion Segmentation Maps)
 └──────────────────┘                        ──► ResNet (Optic Disc & Fovea Localization)
                                                                 │
 ┌──────────────────┐                                            │
 │ Metabolic Data   │ ──► Rule-Based Scoring                     ▼
 │ (HbA1c, BP, Dur) │      (0–9 Points)        ──► [ Decision-Level Fusion ]
 └──────────────────┘                                  (65% Retina + 35% Metabolic)
                                                                 │
                                                                 ▼
                                                  ┌─────────────────────────────┐
                                                  │ • Interactive Streamlit UI  │
                                                  │ • Stratified Recommendation │
                                                  │ • Downloadable PDF Report   │
                                                  └─────────────────────────────┘
```

### Key Models & Modules
* **Classifier (`EfficientNet-B0`):** Fine-tuned via transfer learning with class-balanced sampling to predict DR severity (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative).
* **Lesion Segmentation (`U-Net`):** Generates toggleable segmentation maps for microaneurysms, hemorrhages, hard exudates, and cotton wool spots.
* **Landmark Localization (`ResNet`):** Predicts spatial coordinates of the optic disc and fovea to identify sight-threatening macular involvement.
* **Decision-Level Fusion:** Retinal grade carries a 65% base weighting, complemented by a 35% metabolic risk contribution. Safety overrides ensure Severe (Grade 3) maps to at least High Risk, and Proliferative (Grade 4) strictly enforces an Urgent status.

---

## 📊 Experimental Results & Domain Shift Analysis

The system was evaluated on internal held-out validation data as well as an independent external clinical dataset to assess real-world generalizability:

| Dataset | Evaluation Role | Sample Size | Accuracy | Quadratic Weighted Kappa (QWK) |
| :--- | :--- | :---: | :---: | :---: |
| **APTOS 2019** | Internal Validation (Held-out 20%) | 733 photos | **84.6%** | **0.905** |
| **IDRiD** | External Clinical Validation | 516 photos | **78.8%** | **0.847** |

> **Domain Shift Finding:** Evaluating the classifier on IDRiD without fine-tuning highlighted the impact of cross-camera variances and patient population shifts. Retaining a high QWK of 0.847 demonstrates robust disease ordering preservation while emphasizing the necessity of local calibration before real-world clinical deployment.

---

## 📂 Repository File Structure

```text
Diabetic_Retinopathy_Screening/
│
├── app_premium.py               # Main multi-tab Streamlit dashboard
├── app.py                       # Lightweight launcher script
├── dr_core.py                   # PyTorch models, inference logic & Grad-CAM pipeline
├── premium_theme.py             # Custom glassmorphic CSS styling & UI components
├── report.py                    # PDF report generation engine (ReportLab)
├── risk.py                      # Metabolic risk calculator & decision-fusion rules
├── config.toml                  # Streamlit configuration
├── requirements.txt             # Environment dependencies
│
├── 01_A_data_prep.ipynb         # Data preprocessing, CLAHE pipelines & dataset splits
├── 02_A_train_segmentation_...  # U-Net lesion segmentation & ResNet landmark training
├── 03_B_train_classifier.ipynb  # EfficientNet-B0 fine-tuning on APTOS 2019
├── 04_C_premium_app_launcher... # Colab deployment notebook via tunneling
│
├── LICENSE                      # MIT Open Source License
└── README.md                    # Project documentation
```

---

## 🗄️ Datasets & Pre-Trained Model Checkpoints

Due to GitHub file size limits, raw multi-gigabyte image datasets and PyTorch weight checkpoints (`.pth`/`.pkl`) are hosted on Google Drive:

* 📥 **[Access Full Project Data & Trained Weights (Google Drive)](https://drive.google.com/drive/folders/1ECBI542GevzRRfGTdYaUjdugB2eeNXQL?usp=sharing)**
  * `models/` – EfficientNet-B0 weights, U-Net lesion models, ResNet landmark checkpoints.
  * `data/` – Preprocessed APTOS 2019 and IDRiD image sets.
  * `outputs/` – Evaluation confusion matrices and training loss curves.

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/lambdadesrushti/Diabetic_Retinopathy_Screening.git](https://github.com/lambdadesrushti/Diabetic_Retinopathy_Screening.git)
cd Diabetic_Retinopathy_Screening
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app_premium.py
```

*(Alternatively, run `04_C_premium_app_launcher.ipynb` directly in Google Colab with GPU acceleration enabled).*

---

## 👥 Authors & Collaborators

* **Srushti Lambdade** (PRN: 25030421021) - Data Pipeline, CLAHE Preprocessing, Lesion Segmentation & Landmark Localization
* **Prajakta Suryavanshi** (PRN: 25030421029) - Classifier Architecture, EfficientNet-B0 Fine-Tuning, Grad-CAM & Model Evaluation
* **Sanobar Shaikh** (PRN: 25030421039) - Application Development, Metabolic Risk Scoring, PDF Engine & Deployment
  
### Project Guidance
* **Internal Guides:** Dr. Sahas V Swamy & Dr. Bela Joglekar (Symbiosis Artificial Intelligence Institute, SIU Pune)
* **External Guide:** Mr. Vinayak R Sathe (Co-Founder & Director, StarEast HealthTech)

---

## ⚠️ Clinical Disclaimer
This system is an academic research prototype developed for educational and experimental purposes. It is **not** a certified medical device and is **not intended for autonomous clinical diagnosis or treatment planning**.
