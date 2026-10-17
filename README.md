# 🌱 Smart Crop Yield Prediction & Recommendation System

## 📌 About the Project

The Smart Crop Yield Prediction & Recommendation System is a Machine Learning-based project designed to help farmers make better agricultural decisions. It analyzes soil conditions and environmental factors to recommend suitable crops, predict crop yield, and provide useful agricultural insights.

The project aims to improve farming decisions through Artificial Intelligence, Machine Learning, and Explainable AI (XAI).

## 🎯 Objectives

- Recommend suitable crops based on soil and weather conditions.
- Predict crop yield using historical agricultural data.
- Analyze soil nutrients and pH levels.
- Provide general fertilizer recommendations.
- Identify potential weather-related risks.
- Explain crop recommendations using Explainable AI (SHAP).
- Support data-driven and sustainable agricultural practices.

## ✨ Key Features

🌾 **Crop Recommendation:** Suggests suitable crops based on soil and environmental conditions.

📊 **Crop Yield Prediction:** Estimates crop yield using a trained Machine Learning regression model and suitable historical data.

🧪 **Soil Analysis:** Analyzes Nitrogen (N), Phosphorus (P), Potassium (K), and soil pH values.

🧠 **Explainable AI (SHAP):** Helps users understand which input features influenced a model's prediction.

🌦️ **Weather Intelligence:** Supports weather-based insights and alerts when a weather API is configured.

🌱 **Fertilizer Advisory:** Provides general fertilizer guidance based on available soil and crop information.

📈 **Interactive Dashboard:** Presents predictions and agricultural insights in an easy-to-understand format.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** Scikit-learn
- **Algorithms:** Random Forest, KNN, XGBoost (as implemented)
- **Explainable AI:** SHAP
- **Data Processing:** Pandas, NumPy
- **Model Storage:** Joblib
- **Frontend:** HTML, CSS, JavaScript, Bootstrap
- **Data Visualization:** Chart.js (if implemented)
- **Database:** SQLite / SQLAlchemy (if implemented)

## 📥 Input Parameters

The crop recommendation model may use the following parameters:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Soil pH
- Temperature
- Humidity
- Rainfall

The exact input features depend on the dataset and trained model used in the project.

## ⚙️ How It Works

1. The user enters soil and environmental information.
2. The system validates and preprocesses the input data.
3. The trained Machine Learning model analyzes the input.
4. The system recommends a suitable crop.
5. SHAP explanations help show how input features influenced the recommendation.
6. If a suitable yield model is available, the system estimates crop yield.
7. The dashboard displays the results and available agricultural insights.

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
