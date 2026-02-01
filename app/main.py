import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = FastAPI(title="LSTM Vitals Predictor")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler = None
TIME_STEPS = 10  # Default
EXPECTED_FEATURES = ["heart_rate", "bp", "oxygen_level"]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumes app/main.py, so parent is root. Model is in root/model
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "model", "lstm_model.h5")
SCALER_PATH = os.path.join(ROOT_DIR, "model", "minmax_scaler.joblib")

def load_resources():
    global model, scaler, TIME_STEPS
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            # Try to infer time steps
            if hasattr(model, "input_shape"):
                # input_shape usually (None, time_steps, features)
                ts = model.input_shape[1]
                if ts is not None:
                    TIME_STEPS = ts
            print(f"Model loaded. TIME_STEPS={TIME_STEPS}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e
    else:
        print(f"Model file not found at {MODEL_PATH}")

    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("Scaler loaded.")
        except Exception as e:
            print(f"Failed to load scaler: {e}")
            raise e
    else:
        print(f"Scaler file not found at {SCALER_PATH}")

# Call load on startup (or at module level if simple)
load_resources()

class VitalsInput(BaseModel):
    heart_rate: float
    bp: float
    oxygen_level: float

class PredictionOutput(BaseModel):
    current_hr: float
    predicted_hr: float
    status: str
    risk_level: str

@app.post("/predict", response_model=PredictionOutput)
async def predict_vitals(input_data: VitalsInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded")

    # Prepare DataFrame
    # Logic to map to scaler's expected names
    # Mapping based on previous analysis
    input_dict = {
        "heart_rate": input_data.heart_rate,
        "bp": input_data.bp,
        "oxygen_level": input_data.oxygen_level
    }
    
    # We need to construct a DataFrame that matches what the scaler expects.
    # We'll use the robust logic from the old app.
    
    mapping = {
        'heart_rate': 'Heart rate',
        'bp': 'BP',
        'oxygen_level': 'oxygen'
    }
    
    # Check what the scaler expects
    scaler_features = getattr(scaler, "feature_names_in_", None)
    
    if scaler_features is not None:
        # Create a dict with the correct keys
        # We assume the scaler expects specific names.
        # We construct a single row DF first with our normalized names
        df_norm = pd.DataFrame([input_dict])
        
        # Rename to scaler names
        df_mapped = df_norm.rename(columns=mapping)
        
        # Reorder/Select columns as scaler expects
        # Only keep columns that are in feature_names_in_
        # If some are missing in our input, we might fail, but let's assume the mapping covers it.
        try:
            df_final = df_mapped[list(scaler_features)]
        except KeyError as e:
            # Fallback: maybe the scaler didn't use these names?
            # If we are missing columns, we can't really proceed safely.
            raise HTTPException(status_code=400, detail=f"Scaler expects features: {list(scaler_features)}. Missing: {e}")
            
    else:
        # Fallback if no feature names stored
        df_final = pd.DataFrame([input_dict])
        # Ensure order matches EXPECTED_FEATURES if possible, or just trust the dict order?
        # Ideally we should stick to a strictly defined order.
        df_final = df_final[EXPECTED_FEATURES]

    # Scale the input
    try:
        scaled_values = scaler.transform(df_final) # shape (1, n_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaling failed: {str(e)}")

    # Prepare sequence for LSTM
    # We duplicate the single time point to fill TIME_STEPS
    # shape: (1, TIME_STEPS, n_features)
    seq = np.repeat(scaled_values, TIME_STEPS, axis=0)
    seq = seq.reshape(1, TIME_STEPS, scaled_values.shape[1])

    # Predict
    try:
        prediction = model.predict(seq, verbose=0) # shape (1, n_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Inverse transform
    # prediction contains scaled values for all features.
    # We need to inverse transform to get the real Heart Rate.
    # formatting for inverse_transform
    if scaler_features is not None:
        pred_df = pd.DataFrame(prediction, columns=list(scaler_features))
    else:
        pred_df = pd.DataFrame(prediction, columns=EXPECTED_FEATURES)
        
    try:
        pred_inv = scaler.inverse_transform(pred_df) # shape (1, n_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inverse scaling failed: {str(e)}")
        
    # Extract predicted HR. 
    # We know 'Heart rate' or 'heart_rate' is the target.
    # We need to find which index corresponds to HR.
    # Using the dataframe helps.
    
    # Find the column that represents HR
    hr_col = None
    if "Heart rate" in pred_df.columns:
        hr_col = "Heart rate"
    elif "heart_rate" in pred_df.columns:
        hr_col = "heart_rate"
    
    if hr_col:
        predicted_hr_val = pred_df.iloc[0][hr_col] # Keep it scaled? NO, we want inverse.
        # Wait, pred_inv is a numpy array.
        # We need to map the column index.
        col_list = list(pred_df.columns)
        hr_idx = col_list.index(hr_col)
        predicted_hr_val = pred_inv[0][hr_idx]
    else:
        # Fallback, assume first column?
        predicted_hr_val = pred_inv[0][0]

    # Logic for status
    predicted_hr_val = float(predicted_hr_val)
    
    status = "NORMAL"
    risk_level = "Low Risk"
    
    # Simple logic derived from user's CSV labels or general knowledge
    # CSV had "High Risk" and "Low Risk". 
    # Old app: > 100 is ALERT.
    if predicted_hr_val > 100 or predicted_hr_val < 50:
        status = "ALERT"
        risk_level = "High Risk"
    elif predicted_hr_val > 90:
        risk_level = "Medium Risk" # Just adding some granularity

    return {
        "current_hr": input_data.heart_rate,
        "predicted_hr": round(predicted_hr_val, 2),
        "status": status,
        "risk_level": risk_level
    }

# Mount static files for frontend
# Ensure the directory exists
STATIC_DIR = os.path.join(ROOT_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
