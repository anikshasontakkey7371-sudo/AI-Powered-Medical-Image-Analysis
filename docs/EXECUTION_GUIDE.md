# Execution Guide: AI-Powered Medical Image Analysis

> [!IMPORTANT]
> **Educational Demonstration Only**  
> This project is designed strictly for scientific education, algorithmic learning, and academic exploration. It has **NOT** undergone medical device certification (e.g., FDA 510(k), CE-IVD) and must **NEVER** be utilized for clinical diagnostic evaluations, patient triage, emergency assessment, or healthcare treatment decisions.

---

## 1. System Requirements & Prerequisites

* **Operating System**: Windows 10 or Windows 11 (64-bit)
* **Python**: Version 3.9, 3.10, or 3.11 installed from [python.org](https://www.python.org/downloads/) (ensure *"Add python.exe to PATH"* was selected during installation)
* **Shell**: Windows PowerShell or Windows Terminal

---

## 2. Windows PowerShell Setup (Step-by-Step)

### Step 2.1: Open Windows PowerShell
1. Press the `Windows Key`.
2. Type `PowerShell`.
3. Right-click **Windows PowerShell** and select **Run as administrator** (or open normal PowerShell if you already have execution policy enabled).
4. Navigate to your project directory:
   ```powershell
   cd "C:\Users\Aniksha\OneDrive\Desktop\IIP PROJECTS\AI-Powered Medical Image Analysis"
   ```

### Step 2.2: Enable PowerShell Script Execution (if needed)
If PowerShell blocks virtual environment scripts, run this safe command in your current session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 2.3: Create and Activate a Virtual Environment
Isolating dependencies in a virtual environment prevents conflicts with system packages.

```powershell
# Create the virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```
*(When successfully activated, you will see `(venv)` prepended to your command prompt).*

### Step 2.4: Upgrade pip & Install Required Packages
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Public Dataset Acquisition & Organization

To train the educational model, use an open-access, de-identified chest X-ray dataset.

### Recommended Dataset
* **Chest X-Ray Images (Pneumonia)** by Kermany et al. (Mendeley Data / Kaggle)
  * **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
  * **Source**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) or [Mendeley Data DOI: 10.17632/rscbjbr9sj.2](https://data.mendeley.com/datasets/rscbjbr9sj/2)
  * **Terms**: Free for academic and research purposes with citation.

> [!CAUTION]
> **Privacy Notice**: Never upload or process private, identifiable patient records or un-anonymized clinical scans.

### How to Organize the Files
Extract the downloaded dataset into the `data/raw/` folder in either of two supported structures:

#### Option A: Split Format (Standard Kaggle structure)
```text
data/
└── raw/
    ├── train/
    │   ├── NORMAL/       (contains .jpeg / .png images)
    │   └── PNEUMONIA/    (contains .jpeg / .png images)
    ├── val/              (or validation/)
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

#### Option B: Unified Format (Simplified)
```text
data/
└── raw/
    ├── NORMAL/           (contains .jpeg / .png images)
    └── PNEUMONIA/        (contains .jpeg / .png images)
```
*(If you use Option B, the loader automatically creates an 80/20 train/validation split for you).*

---

## 4. Running the Project Workflows

### Step 4.1: Verify Dataset Readiness
Before training, run the dataset check utility:
```powershell
python main.py check-data
```
This inspects the directory, counts samples per class, and verifies that the directory structure is valid.

---

### Step 4.2: Train the Model
You can choose between the transparent **Custom Baseline CNN** or **MobileNetV2 Transfer Learning**.

#### Training the Custom Baseline CNN (Default)
```powershell
python main.py train --epochs 15 --batch-size 32 --lr 0.001
```

#### Training with Transfer Learning (MobileNetV2)
```powershell
python main.py train --model-type mobilenetv2 --epochs 10 --batch-size 32
```

**What happens during training:**
1. The script automatically sets fixed seeds (`seed=42`) for reproducibility.
2. An augmentation pipeline applies gentle rotations, horizontal flips, and zooms to prevent overfitting.
3. Callbacks monitor `val_loss`:
   * Saves the highest-performing weights to `models/best_model.keras`.
   * Automatically halves the learning rate if validation loss plateaus.
   * Early stopping activates if no improvement is seen for 5 consecutive epochs.
4. After training finishes, the pipeline:
   * Saves training curves (Loss, Accuracy, AUC) to `outputs/training_curves.png`.
   * Generates a confusion matrix heatmap saved to `outputs/confusion_matrix.png`.
   * Prints the classification report (Precision, Recall, F1-score).
   * Saves all raw numerical metrics to `outputs/training_summary.json`.

---

### Step 4.3: Run Predictions on a Single Image
To test inference on an individual X-ray:

```powershell
python main.py predict --image "data/raw/test/PNEUMONIA/person1_virus_6.jpeg"
```

**Prediction Output includes:**
* Predicted diagnostic category (`NORMAL` vs `PNEUMONIA`)
* Model confidence percentage
* Full probability distribution across all classes
* A visual diagnostic card saved to `outputs/prediction_result.png`

---

## 5. Limitations of the Model & AI Ethics

Deep learning models for medical imaging must be understood in context. As a beginner learning about AI in healthcare, keep these essential principles in mind:

1. **Shortcut Learning & Confounders**:
   * Models trained on X-rays can learn spurious features such as hospital radiopaque markers (e.g., "L" or "R" lead markers), patient positioning (supine vs. erect), or image contrast variations rather than genuine lung pathology.
2. **False Negatives & False Positives**:
   * A *false negative* (predicting "NORMAL" when pneumonia is present) can lead to missed diagnosis in clinical practice.
   * A *false positive* can cause unnecessary stress and invasive testing.
   * AI models must never be the sole basis for clinical action.
3. **Distribution Shift**:
   * A model trained on pediatric anterior-posterior (AP) X-rays from one hospital network may completely fail on adult postero-anterior (PA) images or scans from different scanner manufacturers.
4. **Black-Box Limitations**:
   * Standard convolutional networks do not explain *why* an opacity indicates bacterial vs. viral etiology without advanced interpretability tools (e.g., Grad-CAM), which themselves are approximations.
5. **Regulatory & Ethical Boundaries**:
   * Clinical software requires extensive prospective clinical trials, peer review, bias auditing across diverse demographic cohorts, and regulatory clearances (FDA/CE).

---

## 6. Troubleshooting Common Issues

* **PowerShell ExecutionPolicy Error (`File cannot be loaded because running scripts is disabled...`)**:
  * Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current terminal window.
* **`ModuleNotFoundError: No module named 'tensorflow'`**:
  * Ensure your virtual environment is active: `.\venv\Scripts\Activate.ps1`.
  * Confirm installation: `pip list`.
* **Out of Memory (OOM) Errors on GPU/CPU**:
  * Reduce batch size: `python main.py train --batch-size 16`.
* **Image Path Spaces**:
  * Always wrap paths containing spaces in double quotes, e.g.:
    `python main.py predict --image "C:\Path with spaces\sample.jpeg"`
