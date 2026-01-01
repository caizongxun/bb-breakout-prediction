# Usage Guide

## Quick Start

### 1. Download Data

```bash
# Download BTC 15m data
python scripts/download_data.py --symbol BTCUSDT --timeframe 15m

# Download multiple symbols
python scripts/download_data.py --symbols BTCUSDT ETHUSDT --timeframe 15m
```

### 2. Train Models

```bash
# Train LSTM (fastest)
python scripts/train_models.py --symbol BTCUSDT --model lstm --epochs 50

# Train Transformer (best accuracy)
python scripts/train_models.py --symbol BTCUSDT --model transformer --epochs 50

# Train ensemble
python scripts/train_models.py --symbol BTCUSDT --model ensemble
```

### 3. Make Predictions

```bash
# Real-time prediction
python scripts/predict_realtime.py --symbol BTCUSDT --model lstm

# Batch prediction
python scripts/predict_realtime.py --symbol BTCUSDT --model transformer --batch
```

### 4. Backtest Strategy

```bash
# Backtest LSTM model
python scripts/backtest_strategy.py --symbol BTCUSDT --model lstm --threshold 0.6

# Backtest with visualization
python scripts/backtest_strategy.py --symbol BTCUSDT --model transformer --plot
```

## Python API Usage

### Load and Prepare Data

```python
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer

# Load data
loader = DataLoader()
df = loader.load('BTCUSDT', '15m')
print(f"Loaded {len(df)} candles")

# Generate features
engineer = FeatureEngineer()
df = engineer.generate_all_features(df)
df = engineer.create_target(df, forward_window=5)

print(f"Breakout rate: {df['target_breakout'].mean():.2%}")
```

### Train Model

```python
from src.models import LSTMModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Prepare data
X = df[selected_features]
y = df['target_breakout']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences
seq_length = 30
X_seq = np.array([X_scaled[i:i+seq_length] for i in range(len(X_scaled)-seq_length)])
y_seq = y.iloc[seq_length:].values

# Train
model = LSTMModel.create(seq_length, X_seq.shape[2])
model.fit(X_seq, y_seq, epochs=50, batch_size=32)
```

### Make Predictions

```python
# Get latest data
recent_data = X_scaled[-30:]  # Last 30 candles

# Predict
prob = model.predict(recent_data.reshape(1, 30, -1))[0][0]

print(f"Breakout Probability: {prob:.2%}")
print(f"Alert: {'YES' if prob > 0.6 else 'NO'}")
```

### Backtest

```python
from src.evaluation import BacktestStrategy

strategy = BacktestStrategy(model, scaler)
results = strategy.run(df_test, entry_threshold=0.6)

print(results.summary())
print(f"Win Rate: {results.win_rate:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

## Configuration Examples

### Fast Training (LSTM)

```yaml
model:
  type: lstm
  epochs: 30
  batch_size: 32
  
features:
  n_features: 20  # Use fewer features
  auto_engineer: true
```

### High Accuracy (Transformer)

```yaml
model:
  type: transformer
  epochs: 100
  batch_size: 16
  num_blocks: 3
  
features:
  n_features: 50  # More features
  interaction_detection: true
```

### Production (Ensemble)

```yaml
model:
  type: ensemble
  
features:
  auto_engineer: true
  cumulative_importance: 0.90
  interaction_detection: true
```

## Monitoring and Validation

### Check Model Performance

```python
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

print("Accuracy:", (y_pred_binary.flatten() == y_test).mean())
print("AUC:", roc_auc_score(y_test, y_pred.flatten()))
print(classification_report(y_test, y_pred_binary))
```

### Feature Importance

```python
from src.feature_engineering import FeatureEngineer

selected, importance_df = FeatureEngineer.select_features_by_importance(
    X, y, n_features=30
)

importance_df.head(10)  # Top 10 features
```

## Advanced Usage

### Automatic Feature Learning (Autoencoder)

```python
from src.models import AutoencoderFeatureLearner

autoencoder, encoder = AutoencoderFeatureLearner.create(
    seq_length=30,
    n_features=50,
    encoding_dim=16
)

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=50)

# Extract features
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)
```

### Multi-Symbol Learning

```python
# Train on multiple symbols
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
data = []

for symbol in symbols:
    df = loader.load(symbol, '15m')
    df = engineer.generate_all_features(df)
    df = engineer.create_target(df)
    data.append(df)

# Combine data
df_combined = pd.concat(data)
```

### Real-time Monitoring Loop

```python
import time
from datetime import datetime

while True:
    # Get latest candle
    df_latest = loader.load('BTCUSDT', '15m')
    recent = df_latest.iloc[-30:]
    
    # Predict
    recent_scaled = scaler.transform(recent[selected_features])
    prob = model.predict(recent_scaled.reshape(1, 30, -1))[0][0]
    
    if prob > 0.7:  # High confidence
        print(f"[{datetime.now()}] ALERT: Breakout probability {prob:.2%}")
        # Send alert (email, webhook, etc.)
    
    time.sleep(900)  # Check every 15 minutes
```

## Tips for Best Results

1. **Data Quality**: Use at least 1 year of historical data
2. **Feature Engineering**: More features ≠ better, use auto-selection
3. **Model Selection**: Start with LSTM, upgrade to Transformer if needed
4. **Backtesting**: Always backtest before live trading
5. **Regular Retraining**: Retrain monthly with new data
6. **Risk Management**: Never risk more than 2% per trade

## See Also

- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) - Technical details
- [RESULTS.md](RESULTS.md) - Performance benchmarks
- [SETUP.md](SETUP.md) - Installation guide
