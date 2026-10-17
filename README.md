# Smart Crop Yield Prediction & Recommendation System
## BTech Major Project — Explainable AI for Agriculture

A complete web-based agricultural decision-support platform that helps farmers understand crop suitability, estimate yield, analyse soil health, and identify weather-based risks — powered by Machine Learning and SHAP Explainable AI.

---

## 🌾 Features

| Module | Description |
|--------|-------------|
| **Crop Recommendation** | Random Forest Classifier (96.6% test accuracy) recommends the best crop from 22 classes |
| **Explainable AI (SHAP)** | TreeExplainer provides global and local SHAP feature contributions |
| **Yield Prediction** | Random Forest Regressor (R²=0.956) estimates yield in t/ha, kg/ha, quintals/ha |
| **Soil Analysis** | ICAR-based nutrient advisory for N, P, K and pH with crop-specific ranges |
| **Weather Alerts** | Manual input + OpenWeatherMap API with crop-specific risk thresholds |
| **Farmer Dashboard** | Responsive dashboard with stats, recent predictions, and crop distribution chart |
| **Prediction History** | SQLite database stores all predictions; view, filter, delete |
| **PDF Reports** | Downloadable report with inputs, results, SHAP table, and disclaimers |

---

## 🖥️ Prerequisites

- **Python 3.10 or higher** (tested on Python 3.12)
- **pip** (comes with Python)
- **Internet access** (for Bootstrap and Chart.js CDN in the frontend)

> **No GPU required.** Runs comfortably on an ordinary student laptop.

---

## 📁 Project Structure

```
crop yeild/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── gitignore.txt               # Rename to .gitignore before git init
│
├── data/
│   ├── crop_recommendation.csv # Crop recommendation dataset (generated)
│   └── crop_yield.csv          # Yield dataset (generated)
│
├── models/
│   ├── crop_model.joblib       # Trained crop classifier
│   ├── crop_label_encoder.joblib
│   ├── crop_model_metadata.json
│   ├── yield_model.joblib      # Trained yield regressor pipeline
│   ├── yield_model_metadata.json
│   └── yield_model_info.json
│
├── src/
│   ├── generate_datasets.py    # Generates training datasets
│   ├── data_preprocessing.py   # Shared preprocessing utilities
│   ├── train_crop_model.py     # Train crop recommendation model
│   ├── train_yield_model.py    # Train yield prediction model
│   ├── explainability.py       # SHAP explanation module
│   ├── soil_analysis.py        # Soil nutrient advisory
│   ├── weather.py              # Weather risk analysis
│   ├── models_db.py            # SQLAlchemy database models
│   └── report_generator.py     # PDF report generation
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── crop_recommendation.html
│   ├── explainability.html
│   ├── yield_prediction.html
│   ├── soil_analysis.html
│   ├── weather.html
│   ├── history.html
│   └── reports.html
│
├── static/
│   ├── css/style.css           # Custom stylesheet
│   └── js/main.js              # Shared JavaScript utilities
│
├── tests/
│   └── test_system.py          # 65 automated pytest tests
│
└── instance/
    └── predictions.db          # SQLite database (auto-created)
```

---

## ⚙️ Installation

### Step 1 — Find your Python path

Open PowerShell and run:
```powershell
Get-Command python3
```
Note the path (e.g. `C:\Python312\python.exe`).

> If you followed the project setup, Python is at:
> `C:\Users\<YourName>\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

### Step 2 — Navigate to the project folder

```powershell
cd "C:\Users\kapil thakur\Desktop\crop yeild"
```

### Step 3 — Install dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## 🌱 Dataset Preparation

The datasets are already generated. To regenerate them:

```powershell
python src/generate_datasets.py
```

**Data files created:**
- `data/crop_recommendation.csv` — 2200 rows, 22 crops, 7 features
- `data/crop_yield.csv` — 2200 rows, 10 crops, 10 states, 4 seasons

> ⚠️ These datasets are **synthetically generated** using agronomically validated parameter ranges from ICAR and FAO guidelines. They are clearly labelled as synthetic and must not be presented as real survey data. For production use, replace with actual survey datasets.

---

## 🤖 Model Training

### Train Crop Recommendation Model

```powershell
python src/train_crop_model.py
```

Expected output:
- Test Accuracy: ~96.6%
- F1 (weighted): ~96.6%
- Saves: `models/crop_model.joblib`, `crop_label_encoder.joblib`, `crop_model_metadata.json`

### Train Yield Prediction Model

```powershell
python src/train_yield_model.py
```

Expected output:
- MAE: ~2.2 t/ha
- RMSE: ~4.6 t/ha
- R²: ~0.956
- Saves: `models/yield_model.joblib`, `yield_model_metadata.json`, `yield_model_info.json`

> Both models are pre-trained and ready if you downloaded the complete project.

---

## 🔑 Environment Setup

1. Copy the example env file:
```powershell
Copy-Item .env.example .env
```

2. Edit `.env` with a text editor:
```
SECRET_KEY=my-secret-key-change-this
FLASK_DEBUG=false
PORT=5000
OPENWEATHER_API_KEY=   # Optional — get free key at openweathermap.org/api
```

> The application works fully without the weather API key.
> Weather module falls back to manual input mode automatically.

---

## 🚀 Starting the Application

```powershell
python app.py
```

Expected output:
```
INFO: Crop model loaded successfully.
INFO: Yield model loaded successfully.
INFO: Starting Smart Crop System on port 5000
 * Running on http://0.0.0.0:5000
```

Open your browser and go to: **http://localhost:5000**

---

## 🧪 Running Tests

```powershell
python -m pytest tests/ -v
```

Expected: **65 tests passed** in ~10 seconds.

To run a specific test class:
```powershell
python -m pytest tests/test_system.py::TestSHAP -v
python -m pytest tests/test_system.py::TestAPI -v
```

---

## 🌐 API Endpoint Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health and model availability |
| GET | `/api/model-info` | Model metadata, metrics, yield options |
| POST | `/api/crop-recommendation` | Predict best crop from soil/climate inputs |
| POST | `/api/explain-prediction` | SHAP explanations for a prediction |
| GET | `/api/global-importance` | Global SHAP feature importance |
| POST | `/api/yield-prediction` | Predict yield in t/ha, kg/ha, quintals/ha |
| POST | `/api/soil-analysis` | Soil nutrient advisory |
| POST | `/api/weather-analysis` | Weather risk alerts (manual or live API) |
| GET | `/api/prediction-history` | List saved predictions (paginated) |
| GET | `/api/prediction-history/<id>` | Get single prediction record |
| DELETE | `/api/prediction-history/<id>` | Delete a prediction record |
| GET | `/api/dashboard-stats` | Dashboard statistics and recent records |
| POST | `/api/reports/<id>` | Generate and download PDF report |

### Example: Crop Recommendation

```powershell
$body = '{"N":90,"P":42,"K":43,"ph":6.2,"temperature":22,"humidity":65,"rainfall":60}'
Invoke-WebRequest -Uri "http://localhost:5000/api/crop-recommendation" `
  -Method POST -ContentType "application/json" -Body $body
```

### Example: SHAP Explanation

```powershell
Invoke-WebRequest -Uri "http://localhost:5000/api/explain-prediction" `
  -Method POST -ContentType "application/json" -Body $body
```

---

## 🌤️ Weather API Configuration

1. Register at [https://openweathermap.org/api](https://openweathermap.org/api) (free tier available)
2. Copy your API key
3. Add to `.env`: `OPENWEATHER_API_KEY=your_key_here`
4. Restart the application
5. On the Weather page, select **"Live Weather API"** tab and enter a city name (e.g. `Delhi,IN`)

> Without the API key, the **Manual Input** tab is fully functional for weather risk analysis.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `python -m pip install -r requirements.txt` |
| `Model not found` | Run `python src/train_crop_model.py` and `python src/train_yield_model.py` |
| Port 5000 in use | Set `PORT=5001` in `.env` |
| Database error | Delete `instance/predictions.db` — it will be recreated |
| SHAP takes too long | Normal on first load; results are cached for subsequent calls |
| Weather API 401 | Invalid API key — check `OPENWEATHER_API_KEY` in `.env` |
| Weather API 404 | Use format `CityName,CC` (e.g. `Mumbai,IN`) |
| Tests fail | Ensure models are trained first |

---

## 📊 Model Evaluation Summary

| Model | Algorithm | Key Metric |
|-------|-----------|------------|
| Crop Recommendation | Random Forest Classifier | Test Accuracy: 96.59% |
| Crop Recommendation | Random Forest Classifier | CV Accuracy (5-fold): 95.91% ± 0.72% |
| Yield Prediction | Random Forest Regressor | R² = 0.9555, MAE = 2.22 t/ha |

---

## ⚠️ Known Limitations

1. **Synthetic datasets** — Both datasets are synthetically generated. Replace with real survey data for production agricultural use.
2. **SHAP describes model behaviour** — SHAP values explain model decisions, not proven causal agricultural relationships.
3. **Fertilizer doses** — The soil advisory provides preliminary guidance only. Actual fertilizer doses require a certified soil test.
4. **Weather thresholds** — Risk thresholds are indicative ranges from literature, not region-specific calibrated values.
5. **No authentication** — This is a local development application. Add authentication before any multi-user deployment.
6. **Not production-ready** — Security hardening, rate limiting, HTTPS, and performance optimisation are required for production deployment.

---

## 📚 Dataset Sources and References

- Agronomic parameter ranges: ICAR (Indian Council of Agricultural Research) nutrient management guidelines
- Crop-specific thresholds: FAO Crop Production Guidelines
- SHAP methodology: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"

---

## 👨‍🎓 BTech Project Information

- **Project Name:** Smart Crop Yield Prediction & Recommendation System with Explainable AI
- **Technology:** Python, Flask, Scikit-learn, SHAP, SQLite, Bootstrap 5, Chart.js
- **Academic Note:** All predictions carry appropriate disclaimers. This system is for educational and research purposes.

---

*Smart Crop Yield Prediction & Recommendation System — BTech Major Project*
