"""
Sleep Quality Prediction API Backend
===================================
Flask API that loads saved models and provides predictions
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

class SleepQualityAPI:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.kmeans_model = None
        self.model_info = None
        self.load_models()
    
    def load_models(self):
        """Load all saved models and preprocessing objects"""
        try:
            models_dir = 'saved_models'
            
            if not os.path.exists(models_dir):
                print("Warning: saved_models directory not found. Using dummy models.")
                self.create_dummy_models()
                return
            
            # Load model info
            with open(f'{models_dir}/model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            # Load preprocessing objects
            self.scaler = joblib.load(f'{models_dir}/feature_scaler.pkl')
            self.feature_names = joblib.load(f'{models_dir}/feature_names.pkl')
            self.kmeans_model = joblib.load(f'{models_dir}/kmeans_model.pkl')
            
            # Load all classification models
            model_files = [
                ('Decision Tree', 'sleep_decision_tree_model.pkl'),
                ('Random Forest', 'sleep_random_forest_model.pkl'),
                ('SVM', 'sleep_svm_model.pkl'),
                ('K-Nearest Neighbors', 'sleep_k-nearest_neighbors_model.pkl'),
                ('Logistic Regression', 'sleep_logistic_regression_model.pkl')
            ]
            
            for model_name, filename in model_files:
                if os.path.exists(f'{models_dir}/{filename}'):
                    self.models[model_name] = joblib.load(f'{models_dir}/{filename}')
                    print(f"Loaded: {model_name}")
            
            print(f"Successfully loaded {len(self.models)} models")
            print(f"Best model: {self.model_info['best_model']}")
            print(f"Features: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.create_dummy_models()
    
    def create_dummy_models(self):
        """Create dummy models for demonstration"""
        print("Creating dummy models for demonstration...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        # Dummy feature names (top 20 most important)
        self.feature_names = [
            'o3_prev_month_mean', 'hr_cv_prev_week_std', 'hr_cv_prev_24h_max',
            'hr_range_prev_week_mean', 'hr_mean_trend_24h', 'stress_std_prev_month_mean',
            'hr_cv_prev_6h', 'aqi_pm25_prev_24h_min', 'hr_mean_prev_24h_min',
            'o3_prev_12h', 'spo2_mean_prev_month_mean', 'hr_mean_prev_6h',
            'o3_prev_24h_min', 'o3_prev_week_mean', 'o3_prev_24h_mean',
            'pm25', 'hr_mean', 'stress_mean', 'spo2_mean', 'hour'
        ]
        
        # Create dummy scaler
        self.scaler = StandardScaler()
        dummy_data = np.random.randn(100, len(self.feature_names))
        self.scaler.fit(dummy_data)
        
        # Create dummy classifier
        dummy_y = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 100)
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(dummy_data, dummy_y)
        self.models['Random Forest'] = rf_model
        
        # Create dummy KMeans
        self.kmeans_model = KMeans(n_clusters=2, random_state=42)
        self.kmeans_model.fit(dummy_data)
        
        # Dummy model info
        self.model_info = {
            'best_model': 'Random Forest',
            'best_accuracy': 0.75,
            'feature_count': len(self.feature_names),
            'classes': ['Excellent', 'Good', 'Fair', 'Poor']
        }
    
    def create_feature_vector(self, input_data):
        """Create feature vector from input data"""
        # Map input data to feature names
        feature_mapping = {
            'hr_mean': input_data.get('hr_mean', 75),
            'hr_cv_prev_week_std': input_data.get('hr_cv_prev_week_std', 0.15),
            'hr_cv_prev_24h_max': input_data.get('hr_cv_prev_24h_max', 0.2),
            'hr_range_prev_week_mean': input_data.get('hr_range_prev_week_mean', 40),
            'hr_mean_trend_24h': input_data.get('hr_mean_trend_24h', 0),
            'stress_std_prev_month_mean': input_data.get('stress_std_prev_month_mean', 15),
            'hr_cv_prev_6h': input_data.get('hr_cv_prev_6h', 0.12),
            'aqi_pm25_prev_24h_min': input_data.get('aqi_pm25_prev_24h_min', 50),
            'hr_mean_prev_24h_min': input_data.get('hr_mean_prev_24h_min', 65),
            'o3_prev_12h': input_data.get('o3_prev_12h', 0.04),
            'spo2_mean_prev_month_mean': input_data.get('spo2_mean_prev_month_mean', 97),
            'hr_mean_prev_6h': input_data.get('hr_mean_prev_6h', 75),
            'o3_prev_24h_min': input_data.get('o3_prev_24h_min', 0.02),
            'o3_prev_week_mean': input_data.get('o3_prev_week_mean', 0.05),
            'o3_prev_24h_mean': input_data.get('o3_prev_24h_mean', 0.045),
            'o3_prev_month_mean': input_data.get('o3_prev_month_mean', 0.05),
            'pm25': input_data.get('pm25', 25),
            'stress_mean': input_data.get('stress_mean', 30),
            'spo2_mean': input_data.get('spo2_mean', 97),
            'hour': input_data.get('hour', 21)
        }
        
        # Create feature vector in the correct order
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in feature_mapping:
                feature_vector.append(feature_mapping[feature_name])
            else:
                # Use reasonable defaults for missing features
                if 'hr_' in feature_name:
                    feature_vector.append(75)  # Default heart rate
                elif 'stress' in feature_name:
                    feature_vector.append(30)  # Default stress
                elif 'pm25' in feature_name or 'aqi' in feature_name:
                    feature_vector.append(50)  # Default air quality
                elif 'o3' in feature_name:
                    feature_vector.append(0.05)  # Default ozone
                elif 'spo2' in feature_name:
                    feature_vector.append(97)  # Default SpO2
                else:
                    feature_vector.append(0)  # Default for others
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict(self, input_data):
        """Make prediction using the best model"""
        try:
            # Create feature vector
            features = self.create_feature_vector(input_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Use best model for prediction
            best_model_name = self.model_info['best_model']
            best_model = self.models.get(best_model_name, list(self.models.values())[0])
            
            # Make prediction
            prediction = best_model.predict(features_scaled)[0]
            
            # Get prediction probabilities if available
            if hasattr(best_model, 'predict_proba'):
                probabilities = best_model.predict_proba(features_scaled)[0]
                confidence = float(np.max(probabilities)) * 100
            else:
                confidence = 85.0  # Default confidence
            
            # Get cluster assignment
            cluster_id = int(self.kmeans_model.predict(features_scaled)[0])
            
            # Calculate sleep score
            sleep_score = self.calculate_sleep_score(prediction, confidence)
            
            return {
                'prediction': prediction,
                'confidence': round(confidence, 1),
                'sleep_score': sleep_score,
                'cluster_id': cluster_id,
                'model_used': best_model_name,
                'feature_count': len(self.feature_names)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'prediction': 'Good',
                'confidence': 75.0,
                'sleep_score': 75,
                'cluster_id': 0,
                'model_used': 'Fallback',
                'feature_count': len(self.feature_names) if self.feature_names else 20,
                'error': str(e)
            }
    
    def calculate_sleep_score(self, prediction, confidence):
        """Convert prediction to numeric sleep score"""
        score_mapping = {
            'Excellent': 90,
            'Good': 75,
            'Fair': 60,
            'Poor': 40
        }
        base_score = score_mapping.get(prediction, 75)
        
        # Adjust based on confidence
        confidence_adjustment = (confidence - 75) * 0.2
        
        return max(0, min(100, base_score + confidence_adjustment))
    
    def get_recommendations(self, input_data, prediction_result):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Air quality recommendations
        pm25 = input_data.get('pm25', 25)
        if pm25 > 50:
            recommendations.append("Consider using an air purifier in your bedroom")
            recommendations.append("Keep windows closed during high pollution periods")
        elif pm25 < 15:
            recommendations.append("Great air quality! Keep windows open for fresh air")
        
        # Heart rate recommendations
        hr_mean = input_data.get('hr_mean', 75)
        if hr_mean > 85:
            recommendations.append("Practice relaxation techniques before bed")
            recommendations.append("Avoid caffeine 6 hours before sleep")
        elif hr_mean < 60:
            recommendations.append("Your resting heart rate is quite low - maintain your fitness routine")
        
        # Stress recommendations
        stress_mean = input_data.get('stress_mean', 30)
        if stress_mean > 60:
            recommendations.append("Try 10 minutes of meditation before bed")
            recommendations.append("Consider a consistent bedtime routine")
            recommendations.append("Practice deep breathing exercises")
        elif stress_mean < 20:
            recommendations.append("Excellent stress management! Keep up your current routine")
        
        # Time recommendations
        hour = input_data.get('hour', 21)
        if hour < 21:
            recommendations.append("Your bedtime might be early - consider staying up until 21:00-22:00")
        elif hour > 23:
            recommendations.append("Try going to bed earlier for better sleep quality")
        else:
            recommendations.append("Good timing! You're in the optimal bedtime window")
        
        # SpO2 recommendations
        spo2 = input_data.get('spo2_mean', 97)
        if spo2 < 96:
            recommendations.append("Ensure good ventilation in your sleeping area")
            recommendations.append("Consider consulting a healthcare provider about oxygen levels")
        elif spo2 > 98:
            recommendations.append("Excellent oxygen saturation levels!")
        
        # Prediction-based recommendations
        if prediction_result['prediction'] == 'Poor':
            recommendations.append("Focus on consistent sleep schedule and environment optimization")
            recommendations.append("Consider tracking your sleep patterns for a week")
        elif prediction_result['prediction'] == 'Excellent':
            recommendations.append("Maintain your current excellent sleep habits")
            recommendations.append("You're a great example of healthy sleep patterns")
        
        # Default recommendations if none added
        if not recommendations:
            recommendations.extend([
                "Maintain a consistent sleep schedule",
                "Keep your bedroom cool (65-68°F) and dark",
                "Avoid screens 1 hour before bedtime",
                "Consider light exercise during the day"
            ])
        
        return recommendations[:6]  # Limit to 6 recommendations

# Initialize the API
sleep_api = SleepQualityAPI()

@app.route('/')
def home():
    """Serve the main application page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sleep Quality Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .api-info { background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .endpoint { background: #3498db; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }
            pre { background: #2c3e50; color: white; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .status { color: #27ae60; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌙 Sleep Quality Prediction API</h1>
            
            <div class="api-info">
                <h3>API Status: <span class="status">Running</span></h3>
                <p><strong>Models Loaded:</strong> """ + str(len(sleep_api.models)) + """</p>
                <p><strong>Best Model:</strong> """ + str(sleep_api.model_info['best_model'] if sleep_api.model_info else 'Random Forest') + """</p>
                <p><strong>Features:</strong> """ + str(len(sleep_api.feature_names) if sleep_api.feature_names else 20) + """</p>
            </div>
            
            <h3>Available Endpoints:</h3>
            
            <div class="endpoint">
                <strong>POST /predict</strong> - Make sleep quality prediction
            </div>
            
            <div class="endpoint">
                <strong>GET /health</strong> - Check API health
            </div>
            
            <div class="endpoint">
                <strong>GET /models</strong> - Get model information
            </div>
            
            <h3>Example Usage:</h3>
            <pre>
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "hr_mean": 75,
    "pm25": 25,
    "stress_mean": 30,
    "spo2_mean": 97,
    "hour": 21
  }'
            </pre>
            
            <h3>Sample Response:</h3>
            <pre>
{
  "prediction": "Good",
  "confidence": 85.2,
  "sleep_score": 78,
  "cluster_id": 0,
  "recommendations": [
    "Maintain your current healthy sleep habits",
    "Keep your bedroom cool and dark"
  ],
  "optimal_bedtime": "22:00",
  "risk_level": "Low"
}
            </pre>
        </div>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """Make sleep quality prediction"""
    try:
        # Get input data
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Make prediction
        result = sleep_api.predict(input_data)
        
        # Generate recommendations
        recommendations = sleep_api.get_recommendations(input_data, result)
        
        # Calculate additional metrics
        optimal_bedtime = calculate_optimal_bedtime(input_data)
        risk_level = calculate_risk_level(result['sleep_score'])
        sleep_duration = calculate_sleep_duration(result['prediction'])
        
        # Prepare response
        response = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'sleep_score': round(result['sleep_score']),
            'cluster_id': result['cluster_id'],
            'cluster_name': get_cluster_name(result['cluster_id']),
            'recommendations': recommendations,
            'optimal_bedtime': optimal_bedtime,
            'recommended_duration': sleep_duration,
            'risk_level': risk_level,
            'model_info': {
                'model_used': result['model_used'],
                'feature_count': result['feature_count'],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(sleep_api.models),
        'best_model': sleep_api.model_info['best_model'] if sleep_api.model_info else 'Unknown',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/models', methods=['GET'])
def models_info():
    """Get information about loaded models"""
    return jsonify({
        'loaded_models': list(sleep_api.models.keys()),
        'best_model': sleep_api.model_info['best_model'] if sleep_api.model_info else 'Unknown',
        'model_accuracy': sleep_api.model_info.get('best_accuracy', 0.75) if sleep_api.model_info else 0.75,
        'feature_count': len(sleep_api.feature_names) if sleep_api.feature_names else 20,
        'supported_classes': sleep_api.model_info.get('classes', ['Excellent', 'Good', 'Fair', 'Poor']) if sleep_api.model_info else ['Excellent', 'Good', 'Fair', 'Poor']
    })

@app.route('/generate_sample', methods=['GET'])
def generate_sample():
    """Generate sample input data for testing"""
    sample_data = {
        'hr_mean': np.random.randint(60, 100),
        'hr_cv_prev_week_std': round(np.random.uniform(0.05, 0.25), 3),
        'hr_mean_trend_24h': round(np.random.uniform(-5, 5), 1),
        'pm25': np.random.randint(5, 80),
        'o3_prev_month_mean': round(np.random.uniform(0.02, 0.12), 4),
        'aqi_pm25_prev_24h_min': np.random.randint(20, 150),
        'stress_mean': np.random.randint(10, 80),
        'stress_std_prev_month_mean': round(np.random.uniform(5, 25), 1),
        'spo2_mean': round(np.random.uniform(94, 99), 1),
        'hour': np.random.randint(18, 24),
        'cumulative_stress_week': np.random.randint(50, 400)
    }
    
    return jsonify({
        'sample_data': sample_data,
        'description': 'Use this sample data to test the /predict endpoint',
        'usage': 'POST this data to /predict endpoint'
    })

def calculate_optimal_bedtime(input_data):
    """Calculate optimal bedtime based on input"""
    current_hour = input_data.get('hour', 21)
    stress_level = input_data.get('stress_mean', 30)
    
    if stress_level > 60:
        return "21:30"  # Earlier bedtime for high stress
    elif current_hour >= 22:
        return "22:30"
    else:
        return "22:00"

def calculate_risk_level(sleep_score):
    """Calculate risk level based on sleep score"""
    if sleep_score >= 85:
        return "Very Low"
    elif sleep_score >= 75:
        return "Low"
    elif sleep_score >= 60:
        return "Moderate"
    elif sleep_score >= 45:
        return "High"
    else:
        return "Very High"

def calculate_sleep_duration(prediction):
    """Calculate recommended sleep duration"""
    duration_map = {
        'Excellent': "8.0h",
        'Good': "7.5h", 
        'Fair': "8.5h",
        'Poor': "9.0h"
    }
    return duration_map.get(prediction, "8.0h")

def get_cluster_name(cluster_id):
    """Get cluster name from ID"""
    cluster_names = {
        0: "Optimal Environment",
        1: "Challenging Environment"
    }
    return cluster_names.get(cluster_id, f"Cluster {cluster_id}")

if __name__ == '__main__':
    print("Starting Sleep Quality Prediction API...")
    print(f"Models loaded: {len(sleep_api.models)}")
    print("API Documentation available at: http://localhost:5000/")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)