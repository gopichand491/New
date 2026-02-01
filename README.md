# LSTM Patient Monitoring Dashboard

**Overview** ✅
A small Flask app that serves a real-time patient monitoring UI and a `/predict` API backed by an LSTM model (`model/lstm_model.h5`) and a `MinMaxScaler` (`model/minmax_scaler.joblib`).

---

## Quick start 🔧
1. Create & activate the venv (optional but recommended):

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install runtime dependencies:

   ```bash
   python -m pip install -r requirements.txt  # or install individually
   ```

3. Run the app:

   ```bash
   python app.py
   # Open http://127.0.0.1:5000
   ```

---

## Model & scaler details 💡
- Model input shape: **(None, 60, 3)** — i.e., **60 time steps** and **3 features**.
- Output shape: **(None, 3)** — the model predicts the three features.
- Required features (normalized names used inside the app):
  - `heart_rate` → maps to scaler's `Heart rate`
  - `bp` → maps to scaler's `BP`
  - `oxygen_level` → maps to scaler's `oxygen`

The app maps the CSV's normalized column names back to the scaler's original feature names to avoid scikit-learn feature-name validation errors.

**Note:** You may see a warning about scikit-learn version mismatch when unpickling the saved scaler. This is informational; to avoid it, re-fit and re-save the scaler using the current scikit-learn version.

---

## Testing ✅
Run the test suite (uses `pytest`):

```bash
python -m pytest -q
```

Tests included:
- `tests/test_api.py` — verifies `GET /` returns 200 and `GET /predict` returns JSON with keys `current_hr`, `predicted_hr`, and `status`.

---

## Deployment tips ⚠️
- Set `debug=False` and use a production WSGI server (e.g., `waitress-serve` on Windows) for production workloads.

Example (waitress):

```bash
python -m pip install waitress
waitress-serve --port=5000 app:app
```

---

If you'd like, I can add a `requirements.txt` and a `Procfile` for a simple deployment flow. 👇
