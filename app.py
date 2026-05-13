"""
Traffic Speed Predictor - Flask Backend
Simple API for traffic prediction with explanations
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import threading
from model import TrafficLSTM
from explainer import explain_prediction

app = Flask(__name__)
CORS(app)

# Training state (shared across threads)
training_state = {
    "running": False,
    "progress": 0,
    "result": None,
    "error": None
}

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

# Default dataset path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DEFAULT_DATA = os.path.join(DATA_DIR, 'traffic_final_clean.csv')
WORKING_DATA = 'data/traffic.csv'

# Global model instance
trained_model = None

# Copy default dataset to working location on startup
if os.path.exists(DEFAULT_DATA):
    import shutil
    shutil.copy2(DEFAULT_DATA, WORKING_DATA)

# Auto-load trained model on startup
if os.path.exists('saved_models/traffic_model.pth'):
    try:
        trained_model = TrafficLSTM()
        trained_model.load('saved_models/traffic_model.pth')
        print("Pre-trained model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load saved model: {e}")


@app.route('/')
def home():
    return jsonify({
        "message": "Traffic Speed Predictor API",
        "endpoints": {
            "/upload": "POST - Upload CSV traffic data",
            "/train": "POST - Train the LSTM model",
            "/predict": "POST - Get predictions with explanations",
            "/status": "GET - Check model status"
        }
    })


def convert_kaggle_format(df):
    """Auto-convert Kaggle Metro Interstate Traffic Volume format to project format."""
    if 'date_time' in df.columns and 'traffic_volume' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.drop_duplicates(subset='date_time', keep='first').sort_values('date_time').reset_index(drop=True)
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek

        # Greenshields model: speed = Vf * (1 - V/C)
        FREE_FLOW_SPEED, ROAD_CAPACITY = 70.0, 7200.0
        volume_ratio = df['traffic_volume'].clip(upper=ROAD_CAPACITY) / ROAD_CAPACITY
        raw_speed = FREE_FLOW_SPEED * (1.0 - volume_ratio)
        noise = np.random.default_rng(42).normal(0, 1.5, size=len(df))
        df['speed'] = (raw_speed + noise).clip(lower=10.0, upper=75.0).round(1)
        df['volume'] = (df['traffic_volume'] / 100).round(1)
        df['timestamp'] = df['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return df[['timestamp', 'speed', 'volume', 'hour', 'day_of_week']], True
    return df, False


@app.route('/upload', methods=['POST'])
def upload_data():
    """Upload traffic data CSV file (supports project format and Kaggle format)"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file
        filepath = WORKING_DATA
        file.save(filepath)

        # Validate the data
        df = pd.read_csv(filepath)

        # Auto-detect and convert Kaggle Metro Interstate Traffic Volume format
        df, was_converted = convert_kaggle_format(df)
        if was_converted:
            df.to_csv(filepath, index=False)

        required_cols = ['speed', 'volume', 'hour', 'day_of_week']

        # Check if required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({
                "error": f"Missing required columns: {missing_cols}",
                "required": required_cols,
                "found": list(df.columns)
            }), 400

        msg = "Kaggle data converted & loaded" if was_converted else "Data uploaded successfully"

        return jsonify({
            "status": "success",
            "message": msg,
            "rows": len(df),
            "columns": list(df.columns)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _on_progress(info):
    """Callback from model.train_model — receives a dict per epoch."""
    training_state.update({
        "progress": info['percent'],
        "epoch": info['epoch'],
        "total_epochs": info['total_epochs'],
        "train_loss": info['train_loss'],
        "val_loss": info['val_loss'],
        "best_val_loss": info['best_val_loss'],
        "patience": info['patience'],
    })


def _train_worker():
    """Background worker that trains the model."""
    global trained_model, training_state
    try:
        df = pd.read_csv(WORKING_DATA)
        model = TrafficLSTM()

        history = model.train_model(df, progress_callback=_on_progress)

        model.save('saved_models/traffic_model.pth')
        trained_model = model

        metrics = history.get('metrics', {})
        training_state.update({
            "running": False,
            "progress": 100,
            "result": {
                "status": "success",
                "message": "Model trained successfully",
                "epochs": len(history['losses']),
                "final_loss": history['losses'][-1],
                "training_samples": history['samples'],
                "metrics": metrics
            },
            "error": None
        })
    except Exception as e:
        training_state.update({
            "running": False,
            "progress": 0,
            "error": str(e)
        })


@app.route('/train', methods=['POST'])
def train_model():
    """Start model training in the background"""
    global training_state

    if training_state["running"]:
        return jsonify({"error": "Training already in progress"}), 409

    if not os.path.exists(WORKING_DATA):
        return jsonify({"error": "No data uploaded. Please upload data first."}), 400

    training_state = {"running": True, "progress": 0, "epoch": 0, "total_epochs": 80,
                      "train_loss": None, "val_loss": None, "best_val_loss": None,
                      "patience": 0, "result": None, "error": None}
    thread = threading.Thread(target=_train_worker, daemon=True)
    thread.start()

    return jsonify({"status": "started", "message": "Training started"})


@app.route('/train/status', methods=['GET'])
def train_status():
    """Poll training progress with epoch-level detail"""
    if training_state["error"]:
        return jsonify({"status": "error", "error": training_state["error"]}), 500
    if training_state["result"]:
        return jsonify(training_state["result"])
    return jsonify({
        "status": "training",
        "progress": training_state["progress"],
        "epoch": training_state.get("epoch", 0),
        "total_epochs": training_state.get("total_epochs", 80),
        "train_loss": training_state.get("train_loss"),
        "val_loss": training_state.get("val_loss"),
        "best_val_loss": training_state.get("best_val_loss"),
        "patience": training_state.get("patience", 0),
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Make traffic speed prediction with explanations"""
    global trained_model

    try:
        # Check if model is loaded
        if trained_model is None:
            # Try to load saved model
            if os.path.exists('saved_models/traffic_model.pth'):
                trained_model = TrafficLSTM()
                trained_model.load('saved_models/traffic_model.pth')
            else:
                return jsonify({"error": "Model not trained. Please train the model first."}), 400

        # Get input data
        data = request.json
        if 'data' not in data:
            return jsonify({"error": "No input data provided"}), 400

        input_data = np.array(data['data'])

        # Validate input shape
        if input_data.shape[1] != 4:
            return jsonify({
                "error": "Input must have 4 features: speed, volume, hour, day_of_week"
            }), 400

        # Make prediction
        prediction = trained_model.predict(input_data)

        # Get explanation
        explanation = explain_prediction(trained_model, input_data)

        return jsonify({
            "prediction": round(prediction, 1),
            "unit": "mph",
            "explanation": explanation,
            "temporal_attention": explanation.get("temporal_attention", None)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """Check model and data status"""
    data_exists = os.path.exists(WORKING_DATA)
    model_exists = os.path.exists('saved_models/traffic_model.pth')

    data_info = None
    if data_exists:
        df = pd.read_csv(WORKING_DATA)
        data_info = {
            "rows": len(df),
            "columns": list(df.columns)
        }

    return jsonify({
        "data_uploaded": data_exists,
        "model_trained": model_exists,
        "model_loaded": trained_model is not None,
        "data_info": data_info
    })


@app.route('/data-stats', methods=['GET'])
def data_stats():
    """Get statistics about the loaded dataset"""
    try:
        if not os.path.exists(WORKING_DATA):
            return jsonify({"error": "No data loaded"}), 400

        df = pd.read_csv(WORKING_DATA)
        return jsonify({
            "rows": int(len(df)),
            "speed": {"min": float(round(df['speed'].min(), 1)), "max": float(round(df['speed'].max(), 1)), "mean": float(round(df['speed'].mean(), 1))},
            "volume": {"min": float(round(df['volume'].min(), 1)), "max": float(round(df['volume'].max(), 1)), "mean": float(round(df['volume'].mean(), 1))},
            "date_range": {"start": str(df['timestamp'].iloc[0]), "end": str(df['timestamp'].iloc[-1])}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate-sample', methods=['POST'])
def generate_sample():
    """Generate realistic traffic data modeled after Kaggle Metro Interstate patterns."""
    try:
        num_hours = request.json.get('hours', 168)  # Default: 1 week

        rng = np.random.default_rng(42)
        timestamps = pd.date_range(start='2016-01-01', periods=num_hours, freq='h')

        data = []
        prev_speed = 60.0  # for temporal continuity

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            day = ts.dayofweek
            month = ts.month

            # --- Volume model (modeled after I-94 patterns) ---
            # Base hourly profile (vehicles/hr, scaled /100 for model input)
            hour_volume_profile = {
                0: 3, 1: 2, 2: 1.5, 3: 1.2, 4: 1.5, 5: 3,
                6: 8, 7: 20, 8: 35, 9: 28, 10: 18, 11: 16,
                12: 18, 13: 17, 14: 18, 15: 22, 16: 35, 17: 42,
                18: 32, 19: 20, 20: 14, 21: 10, 22: 7, 23: 5
            }
            base_vol = hour_volume_profile[hour]

            # Weekend: less commuter traffic, more midday
            if day >= 5:
                if 7 <= hour <= 9 or 16 <= hour <= 18:
                    base_vol *= 0.45  # No rush hour on weekends
                elif 10 <= hour <= 15:
                    base_vol *= 1.3   # More midday activity

            # Seasonal variation (more volume in summer)
            seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 1) / 12 - np.pi / 3)
            base_vol *= seasonal_factor

            # Random variation
            volume = max(0.5, base_vol + rng.normal(0, base_vol * 0.15))

            # --- Speed model (Greenshields + noise + temporal continuity) ---
            free_flow = 70.0
            capacity_vol = 50.0  # in scaled units (/100)
            speed_from_volume = free_flow * (1.0 - min(volume / capacity_vol, 0.95))

            # Weather/random events (occasional drops)
            if rng.random() < 0.03:
                speed_from_volume *= rng.uniform(0.6, 0.85)  # Bad weather / incident

            # Temporal continuity — speed doesn't jump wildly hour to hour
            target_speed = speed_from_volume + rng.normal(0, 2.0)
            speed = 0.7 * target_speed + 0.3 * prev_speed  # Smoothing
            speed = max(12.0, min(72.0, speed))
            prev_speed = speed

            data.append({
                'timestamp': ts,
                'speed': round(speed, 1),
                'volume': round(volume, 1),
                'hour': hour,
                'day_of_week': day
            })

        df = pd.DataFrame(data)
        df.to_csv(WORKING_DATA, index=False)

        return jsonify({
            "status": "success",
            "message": f"Generated {num_hours} hours of realistic traffic data",
            "rows": len(df)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting Traffic Prediction Server...")
    print("API Documentation: http://localhost:5000")
    try:
        from waitress import serve
        print("Using Waitress WSGI server")
        serve(app, host='0.0.0.0', port=5000)
    except ImportError:
        app.run(debug=False, port=5000, threaded=True)
