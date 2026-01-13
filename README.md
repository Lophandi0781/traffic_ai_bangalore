# Bangalore Traffic AI Predictor

## Run steps (Windows)

### 1) Create venv
python -m venv .venv
.venv\Scripts\activate

### 2) Install dependencies
pip install -r training\requirements.txt
pip install -r backend\requirements.txt
pip install -r frontend\requirements.txt

### 3) Train model
python training\train.py

### 4) Run API
uvicorn backend.app:app --reload --port 8000

### 5) Run UI
streamlit run frontend\streamlit_app.py
