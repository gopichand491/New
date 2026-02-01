@echo off
echo Activating virtual environment...
call .venv\Scripts\activate

echo Starting LSTM Vitals Predictor...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
