
@echo off
echo ==========================================
echo Setting up Fraud Detection System Environment
echo ==========================================

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo ==========================================
echo Running ML Pipeline
echo ==========================================

echo 1. Generating Data...
python src/data_ingestion.py

echo 2. Training Models...
python src/train.py

echo 3. Generating EDA Reports...
python notebooks/eda.py

echo ==========================================
echo Starting API Server
echo ==========================================
echo API will be available at http://127.0.0.1:8000
echo Frontend will be available at http://127.0.0.1:8000/
echo Swagger Docs at http://127.0.0.1:8000/docs

uvicorn api.main:app --reload
