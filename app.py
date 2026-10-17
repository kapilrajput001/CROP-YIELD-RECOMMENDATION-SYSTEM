"""
Smart Crop Yield Prediction & Recommendation System
Main Flask Application

Start with:  python app.py
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone

from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_cors import CORS
from dotenv import load_dotenv, dotenv_values

# ---- Project imports ----
sys.path.insert(0, os.path.dirname(__file__))
from src.models_db import db, PredictionHistory
from src.data_preprocessing import validate_crop_rec_input, CROP_REC_FEATURES
from src.soil_analysis import analyse_soil
from src.weather import fetch_weather_and_analyse, analyse_manual_weather
from src.report_generator import generate_report

# ---- Load environment (override=True so a re-load always wins) ----
load_dotenv(override=True)

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Create app ----
app = Flask(__name__)
CORS(app)

# ---- Config ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'predictions.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB

os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)

db.init_app(app)

# ---- Lazy-loaded ML models ----
_crop_model = None
_crop_le = None
_crop_meta = None
_yield_pipeline = None
_yield_meta = None
_yield_info = None


def _load_crop_model():
    global _crop_model, _crop_le, _crop_meta
    if _crop_model is not None:
        return True
    import joblib
    model_path = os.path.join(BASE_DIR, "models", "crop_model.joblib")
    le_path = os.path.join(BASE_DIR, "models", "crop_label_encoder.joblib")
    meta_path = os.path.join(BASE_DIR, "models", "crop_model_metadata.json")
    if not os.path.exists(model_path):
        logger.error("Crop model not found. Run: python src/train_crop_model.py")
        return False
    _crop_model = joblib.load(model_path)
    _crop_le = joblib.load(le_path)
    with open(meta_path) as f:
        _crop_meta = json.load(f)
    logger.info("Crop model loaded successfully.")
    return True


def _load_yield_model():
    global _yield_pipeline, _yield_meta, _yield_info
    if _yield_pipeline is not None:
        return True
    import joblib
    pipeline_path = os.path.join(BASE_DIR, "models", "yield_model.joblib")
    meta_path = os.path.join(BASE_DIR, "models", "yield_model_metadata.json")
    info_path = os.path.join(BASE_DIR, "models", "yield_model_info.json")
    if not os.path.exists(pipeline_path):
        logger.error("Yield model not found. Run: python src/train_yield_model.py")
        return False
    _yield_pipeline = joblib.load(pipeline_path)
    with open(meta_path) as f:
        _yield_meta = json.load(f)
    with open(info_path) as f:
        _yield_info = json.load(f)
    logger.info("Yield model loaded successfully.")
    return True


# ================================================================
# ROUTES — Pages
# ================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/crop-recommendation")
def crop_recommendation_page():
    return render_template("crop_recommendation.html")


@app.route("/explainability")
def explainability_page():
    return render_template("explainability.html")


@app.route("/yield-prediction")
def yield_prediction_page():
    return render_template("yield_prediction.html")


@app.route("/soil-analysis")
def soil_analysis_page():
    return render_template("soil_analysis.html")


@app.route("/weather")
def weather_page():
    return render_template("weather.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/reports")
def reports_page():
    return render_template("reports.html")


# ================================================================
# Utility routes — suppress browser-extension noise
# ================================================================

@app.route("/favicon.ico")
def favicon():
    """Return an empty 204 so browsers stop logging 404s."""
    return Response(status=204)


@app.route("/hybridaction/<path:subpath>")
def suppress_tracker(subpath):
    """Silently discard Zybra tracker requests injected by a browser extension."""
    return Response(status=204)


# ================================================================
# API — Health & Info
# ================================================================

@app.route("/api/health")
def health():
    crop_ok = _crop_model is not None or os.path.exists(
        os.path.join(BASE_DIR, "models", "crop_model.joblib"))
    yield_ok = _yield_pipeline is not None or os.path.exists(
        os.path.join(BASE_DIR, "models", "yield_model.joblib"))
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "crop_model_available": crop_ok,
        "yield_model_available": yield_ok,
        "weather_api_configured": bool(os.environ.get("OPENWEATHER_API_KEY", "")),
    })


@app.route("/api/model-info")
def model_info():
    info = {}
    meta_path = os.path.join(BASE_DIR, "models", "crop_model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            info["crop_model"] = json.load(f)
    yield_meta_path = os.path.join(BASE_DIR, "models", "yield_model_metadata.json")
    if os.path.exists(yield_meta_path):
        with open(yield_meta_path) as f:
            info["yield_model"] = json.load(f)
    yield_info_path = os.path.join(BASE_DIR, "models", "yield_model_info.json")
    if os.path.exists(yield_info_path):
        with open(yield_info_path) as f:
            info["yield_options"] = json.load(f)
    return jsonify(info)


# ================================================================
# API — Crop Recommendation
# ================================================================

@app.route("/api/crop-recommendation", methods=["POST"])
def crop_recommendation():
    if not _load_crop_model():
        return jsonify({"error": "Crop model not available. Run training script."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    validation = validate_crop_rec_input(data)
    if not validation["valid"]:
        return jsonify({"error": "Validation failed.", "details": validation["errors"]}), 422

    import pandas as pd
    import numpy as np

    input_df = pd.DataFrame([{f: float(data[f]) for f in CROP_REC_FEATURES}])
    pred_idx = int(_crop_model.predict(input_df)[0])
    predicted_crop = _crop_le.inverse_transform([pred_idx])[0]
    probas = _crop_model.predict_proba(input_df)[0]

    # Top 5 predictions
    top5_idx = np.argsort(probas)[::-1][:5]
    top5 = [
        {
            "crop": _crop_le.inverse_transform([i])[0],
            "probability": round(float(probas[i]), 4),
        }
        for i in top5_idx
    ]

    record = PredictionHistory(
        prediction_type="crop",
        input_data=json.dumps({f: float(data[f]) for f in CROP_REC_FEATURES}),
        recommended_crop=predicted_crop,
        crop_probability=round(float(probas[pred_idx]), 4),
        model_version=_crop_meta.get("model_type", "RandomForest"),
        model_accuracy=_crop_meta.get("metrics", {}).get("test_accuracy"),
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "prediction_id": record.id,
        "recommended_crop": predicted_crop,
        "model_score": round(float(probas[pred_idx]), 4),
        "top_5_crops": top5,
        "model_accuracy": _crop_meta.get("metrics", {}).get("test_accuracy"),
        "disclaimer": (
            "Model score reflects how strongly input conditions match the training "
            "patterns for this crop. It is not a guaranteed real-world outcome."
        ),
    })


# ================================================================
# API — SHAP Explanation
# ================================================================

@app.route("/api/explain-prediction", methods=["POST"])
def explain_prediction_api():
    if not _load_crop_model():
        return jsonify({"error": "Crop model not available."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    # We need crop features; prediction_id is optional (to update history)
    validation = validate_crop_rec_input(data)
    if not validation["valid"]:
        return jsonify({"error": "Validation failed.", "details": validation["errors"]}), 422

    from src.explainability import explain_prediction, explain_top_crops

    try:
        local_exp = explain_prediction(data)
        top3 = explain_top_crops(data, top_n=3)
    except Exception as e:
        logger.exception("SHAP explanation failed")
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500

    # Optionally update the history record
    pred_id = data.get("prediction_id")
    if pred_id:
        rec = db.session.get(PredictionHistory, int(pred_id))
        if rec:
            rec.shap_summary = json.dumps(local_exp)
            db.session.commit()

    return jsonify({
        "local_explanation": local_exp,
        "top3_crop_explanations": top3,
    })


@app.route("/api/global-importance")
def global_importance():
    if not _load_crop_model():
        return jsonify({"error": "Crop model not available."}), 503
    from src.explainability import get_global_feature_importance
    try:
        result = get_global_feature_importance()
        return jsonify(result)
    except Exception as e:
        logger.exception("Global importance failed")
        return jsonify({"error": str(e)}), 500


# ================================================================
# API — Yield Prediction
# ================================================================

@app.route("/api/yield-prediction", methods=["POST"])
def yield_prediction():
    if not _load_yield_model():
        return jsonify({"error": "Yield model not available. Run training script."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    required = ["crop", "state", "season", "year", "rainfall_mm", "temperature_c"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 422

    import pandas as pd
    import numpy as np

    try:
        input_df = pd.DataFrame([{
            "crop": str(data["crop"]),
            "state": str(data["state"]),
            "season": str(data["season"]),
            "year": int(data["year"]),
            "area_ha": float(data.get("area_ha", 1000)),
            "rainfall_mm": float(data["rainfall_mm"]),
            "temperature_c": float(data["temperature_c"]),
        }])
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input values: {e}"}), 422

    try:
        predicted_yield = float(_yield_pipeline.predict(input_df)[0])
    except Exception as e:
        logger.exception("Yield prediction failed")
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    predicted_yield = max(0.0, round(predicted_yield, 3))

    area = float(data.get("area_ha", 0))
    estimated_production = round(predicted_yield * area, 2) if area > 0 else None

    # Conversion factors
    yield_t_ha = predicted_yield
    yield_kg_ha = round(predicted_yield * 1000, 1)
    yield_q_ha = round(predicted_yield * 10, 2)   # quintals (1t = 10q)

    record = PredictionHistory(
        prediction_type="yield",
        input_data=json.dumps({k: str(v) for k, v in data.items()}),
        predicted_yield=yield_t_ha,
        yield_unit="t/ha",
        cultivated_area=area if area > 0 else None,
        estimated_production=estimated_production,
        model_version=_yield_meta.get("model_type", "RandomForest"),
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "prediction_id": record.id,
        "crop": data["crop"],
        "state": data["state"],
        "season": data["season"],
        "year": data["year"],
        "predicted_yield_t_per_ha": yield_t_ha,
        "predicted_yield_kg_per_ha": yield_kg_ha,
        "predicted_yield_q_per_ha": yield_q_ha,
        "cultivated_area_ha": area if area > 0 else None,
        "estimated_production_tonnes": estimated_production,
        "model_metrics": _yield_meta.get("metrics", {}),
        "disclaimer": (
            "Yield prediction is based on a regression model trained on historical "
            "data. Actual yields depend on local conditions, variety, management, "
            "and weather. Do not treat this as a guaranteed harvest quantity."
        ),
    })


# ================================================================
# API — Soil Analysis
# ================================================================

@app.route("/api/soil-analysis", methods=["POST"])
def soil_analysis_api():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    result = analyse_soil(data)
    if "error" in result:
        return jsonify(result), 422

    # Optionally update history record
    pred_id = data.get("prediction_id")
    if pred_id:
        try:
            rec = db.session.get(PredictionHistory, int(pred_id))
            if rec:
                rec.soil_advisory = json.dumps(result)
                db.session.commit()
        except Exception:
            pass

    return jsonify(result)


# ================================================================
# API — Weather Analysis
# ================================================================

@app.route("/api/weather-analysis", methods=["POST"])
def weather_analysis():
    # Re-load .env so a key added after startup is always picked up
    load_dotenv(override=True)

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    mode = data.get("mode", "manual")

    if mode == "location":
        location = str(data.get("location", ""))
        if not location:
            return jsonify({"error": "Location required."}), 422
        crop = str(data.get("crop", "General"))
        result = fetch_weather_and_analyse(location, crop)
    else:
        result = analyse_manual_weather(data)

    if "error" in result:
        return jsonify(result), 422

    pred_id = data.get("prediction_id")
    if pred_id:
        try:
            rec = db.session.get(PredictionHistory, int(pred_id))
            if rec:
                rec.weather_advisory = json.dumps(result)
                db.session.commit()
        except Exception:
            pass

    return jsonify(result)


# ================================================================
# API — Prediction History
# ================================================================

@app.route("/api/prediction-history")
def prediction_history():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    per_page = min(per_page, 100)

    query = (
        PredictionHistory.query
        .order_by(PredictionHistory.timestamp.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    return jsonify({
        "records": [r.to_dict() for r in query.items],
        "total": query.total,
        "page": page,
        "pages": query.pages,
        "per_page": per_page,
    })


@app.route("/api/prediction-history/<int:record_id>")
def prediction_history_detail(record_id: int):
    rec = db.session.get(PredictionHistory, record_id)
    if not rec:
        return jsonify({"error": "Record not found."}), 404
    return jsonify(rec.to_dict())


@app.route("/api/prediction-history/<int:record_id>", methods=["DELETE"])
def delete_prediction(record_id: int):
    rec = db.session.get(PredictionHistory, record_id)
    if not rec:
        return jsonify({"error": "Record not found."}), 404
    db.session.delete(rec)
    db.session.commit()
    return jsonify({"message": "Record deleted."})


@app.route("/api/dashboard-stats")
def dashboard_stats():
    total = PredictionHistory.query.count()
    crop_recs = PredictionHistory.query.filter_by(prediction_type="crop").count()
    yield_preds = PredictionHistory.query.filter_by(prediction_type="yield").count()

    recent = (
        PredictionHistory.query
        .order_by(PredictionHistory.timestamp.desc())
        .limit(5)
        .all()
    )

    # Crop distribution (last 50 recommendations)
    last_50 = (
        PredictionHistory.query
        .filter_by(prediction_type="crop")
        .filter(PredictionHistory.recommended_crop.isnot(None))
        .order_by(PredictionHistory.timestamp.desc())
        .limit(50)
        .all()
    )
    crop_counts = {}
    for r in last_50:
        c = r.recommended_crop
        crop_counts[c] = crop_counts.get(c, 0) + 1

    return jsonify({
        "total_predictions": total,
        "crop_recommendations": crop_recs,
        "yield_predictions": yield_preds,
        "recent_predictions": [r.to_dict() for r in recent],
        "crop_distribution": crop_counts,
    })


# ================================================================
# API — Report Generation
# ================================================================

@app.route("/api/reports/<int:record_id>", methods=["POST"])
def generate_report_api(record_id: int):
    rec = db.session.get(PredictionHistory, record_id)
    if not rec:
        return jsonify({"error": "Record not found."}), 404

    try:
        pdf_bytes = generate_report(rec.to_dict())
    except Exception as e:
        logger.exception("Report generation failed")
        return jsonify({"error": f"Report generation failed: {e}"}), 500

    import io
    buf = io.BytesIO(pdf_bytes)
    buf.seek(0)

    filename = f"crop_report_{record_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


# ================================================================
# Error handlers
# ================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Request too large."}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ================================================================
# App startup
# ================================================================

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        logger.info("Database tables created/verified.")
        # Pre-load models at startup
        _load_crop_model()
        _load_yield_model()

    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Smart Crop System on port {port}")
    app.run(debug=debug_mode, port=port, host="0.0.0.0")
