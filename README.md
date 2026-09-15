# Freight Rate Prediction Challenge - Run Instructions

## Quick Start & Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-folder-name>
```
### 2. Create and Activate a Virtual Environment
macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (Command Prompt / PowerShell):
```bash
python -m venv venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Pipeline Execution
Run the scripts in the following sequential order to process data, train, and generate predictions:

#### 1. Dataset Cleaning & Exploration:
```bash
python preprocessing.py
```

#### 2. Feature Engineering:
```bash
python feature_engineering.py
```

#### 3. Model Evaluation & Feature Importance:
```bash
python model.py
```
#### 4. Validation Predictions:
```bash
python validation.py
```

#### 5. December Predictions:
```bash
python predict_december.py
```
