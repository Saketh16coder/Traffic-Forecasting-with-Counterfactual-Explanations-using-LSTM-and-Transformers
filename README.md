# Explainable Traffic Forecasting with Counterfactual Explanations

An AI-powered traffic speed prediction system using an **LSTM + Transformer hybrid** neural network with explainable AI features including counterfactual what-if scenarios, temporal attention visualization, and feature importance analysis.



## Dataset

This project uses the **Metro Interstate Traffic Volume** dataset from Kaggle:

**Download:** https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume

The dataset contains **48,204 hourly records** of I-94 Westbound traffic volume from MN DoT ATR station 301 (between Minneapolis and St Paul, MN), collected from 2012-2018.

### Quick Setup

**Option A — Upload via the web UI (recommended):**
Simply upload the raw Kaggle CSV through the frontend — it auto-detects the format and converts it on the fly.

**Option B — Preprocess before running:**
```bash
# 1. Download Metro_Interstate_Traffic_Volume.csv from Kaggle
# 2. Place it in the project root
python scripts/prepare_kaggle_data.py Metro_Interstate_Traffic_Volume.csv
# 3. Start the backend as usual
```

### How the Conversion Works

The Kaggle dataset has `traffic_volume` but no `speed`. Speed is derived using the **Greenshields traffic flow model**, a standard traffic-engineering formula:

```
speed = free_flow_speed x (1 - volume / road_capacity)
```

Parameters: `free_flow_speed = 70 mph`, `road_capacity = 7200 veh/hr`

## Project Structure

```
traffic-predictor/
├── backend/
│   ├── app.py             
│   ├── model.py            
│   ├── explainer.py        
│   └── requirements.txt   
├── frontend/
│   ├── index.html          
│   ├── style.css          
│   └── script.js           
├── scripts/
│   ├── prepare_kaggle_data.py  
│   └── train_fresh.py          
├── data/
│   └── traffic_final_clean.csv 
└── README.md
```

## Features

### Core
- **CSV Upload** — Upload traffic data or generate realistic sample data
- **Kaggle Auto-Convert** — Upload raw Kaggle CSV, auto-detected and converted
- **LSTM + Transformer Training** — Hybrid neural network with real-time epoch progress
- **Real-time Predictions** — Predict traffic speed for the next hour

### Explainable AI (XAI)
- **Feature Importance** — Perturbation-based analysis showing % impact of each feature
- **Temporal Attention** — Transformer attention weights showing which past hours matter most
- **What-If Scenarios** — 6 counterfactual scenarios (volume changes, speed changes, rush hour, weekend)
- **Natural Language Explanations** — Human-readable prediction summaries with recommendations

### Training Pipeline
- **Train/Validation Split** — 85/15 split with proper holdout evaluation
- **OneCycleLR Scheduler** — Fast convergence with cosine annealing
- **Early Stopping** — Patience-based stopping to prevent overfitting
- **Live Progress** — Real-time epoch, loss, and validation loss displayed during training
- **Accuracy Metrics** — MAE, RMSE, R², accuracy within 3/5 mph reported after training

## Model Architecture

```
Input (12 hours x 4 features)
        |
   LSTM Layer 1 (64 units, dropout 0.15)
        |
   LSTM Layer 2 (64 units, dropout 0.15)
        |
   Positional Encoding (sinusoidal)
        |
   Transformer Encoder (2 layers, 2-head attention)
        |
   Dense Layer --> 1 output (predicted speed in mph)
```

| Component | Details |
|-----------|---------|
| LSTM | 2 layers, 64 hidden units, captures sequential patterns |
| Positional Encoding | Sinusoidal, adds temporal position awareness |
| Transformer | 2 encoder layers, 2-head self-attention, 256-dim feedforward |
| Optimizer | AdamW (lr=0.002, weight_decay=1e-4) |
| Scheduler | OneCycleLR (cosine annealing, 30% warmup) |
| Loss | HuberLoss (delta=1.0, robust to outliers) |
| Input | 12 time steps x 4 features |
| Output | 1 value (next-hour speed in mph) |

## Model Accuracy

Validated on 15% holdout set (~6,000 samples):

| Metric | Value |
|--------|-------|
| MAE | 2.01 mph |
| RMSE | 2.77 mph |
| R² | 0.9788 |
| Accuracy (within 3 mph) | 78.2% |
| Accuracy (within 5 mph) | 93.8% |
| Training Loss (Huber) | 0.0022 |
| Early Stopped At | Epoch 78/80 |

## Input Data Format

### Required CSV Columns

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| `timestamp` | Date/time of measurement | datetime | 2024-01-01 08:00 |
| `speed` | Traffic speed in mph | float | 55.5 |
| `volume` | Traffic volume (scaled /100) | float | 15.0 |
| `hour` | Hour of day (0-23) | int | 8 |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) | int | 1 |


## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available endpoints |
| `/status` | GET | Check data upload and model training status |
| `/upload` | POST | Upload CSV traffic data (supports Kaggle format) |
| `/train` | POST | Start model training (async, returns immediately) |
| `/train/status` | GET | Poll training progress (epoch, loss, val_loss) |
| `/predict` | POST | Get prediction with full XAI explanations |
| `/data-stats` | GET | Get statistics about the loaded dataset |
| `/generate-sample` | POST | Generate realistic synthetic traffic data |

### Predict Response Example

```json
{
  "prediction": 56.4,
  "unit": "mph",
  "explanation": {
    "current_prediction": 56.4,
    "feature_importance": {
      "speed": 26.5,
      "volume": 8.8,
      "hour": 64.7,
      "day_of_week": 0.0
    },
    "explanation": "The predicted traffic speed is 56.4 mph, indicating free-flowing conditions...",
    "counterfactual": {
      "scenarios": [
        {
          "description": "If traffic volume dropped significantly (-70%)",
          "original_prediction": 56.4,
          "new_prediction": 58.2,
          "change": 1.8
        }
      ]
    },
    "temporal_attention": {
      "weights": [0.077, 0.076, ...],
      "labels": ["t-11", "t-10", ...],
      "description": "The model attended most to 6 hours ago (weight: 9.9%)..."
    }
  }
}
```

## Installation

### Backend

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

The API will be available at `http://localhost:5000`

### Frontend

```bash
cd frontend
python -m http.server 8080
```

Then open `http://localhost:8080`

## Usage

1. **Upload Data** — Drop a Kaggle CSV or click "Generate Sample" for testing
2. **Train Model** — Click "Train Model" and watch real-time epoch progress (loss, val_loss)
3. **Predict** — Enter speed, volume, hour, day and click "Predict Traffic Speed"
4. **Analyze Results**:
   - Predicted speed with condition badge (Free Flowing / Moderate / Congested)
   - Input summary cards (avg speed, volume, hour, day)
   - Feature importance bar chart
   - Temporal attention bar chart
   - Natural language explanation
   - 6 what-if scenario cards with impact values

## Technologies

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.8+, Flask 3.0, Flask-CORS |
| ML Framework | PyTorch 2.1 |
| Model | LSTM + Transformer Encoder (hybrid) |
| Preprocessing | scikit-learn (MinMaxScaler), Pandas, NumPy |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Charts | Chart.js |
| Fonts | Syne (display), DM Sans (body) |

## License

MIT License
