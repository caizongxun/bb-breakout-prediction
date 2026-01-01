# BB Breakout Prediction Model

Machine Learning system for predicting Bollinger Band breakout movements in cryptocurrency markets with automatic feature engineering and real-time inference.

## Overview

This project aims to predict major volatility expansions (BB breakouts) in crypto markets 3-10 candles ahead of time using:

- **LSTM Neural Networks** for sequential pattern recognition
- **Transformer Models** with attention mechanisms
- **XGBoost** for gradient boosting classification
- **AutoML (AutoGluon)** for automatic model selection
- **Automatic Feature Engineering** to discover optimal feature combinations

## Quick Start

```bash
# Clone repository
git clone https://github.com/caizongxun/bb-breakout-prediction.git
cd bb-breakout-prediction

# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py --symbols BTCUSDT

# Train model
python scripts/train_models.py --model lstm

# Make predictions
python scripts/predict_realtime.py --symbol BTCUSDT
```

## Data Source

- **HuggingFace Dataset**: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
- **Supported Timeframes**: 15m, 1h
- **Data Format**: Parquet (OHLCV)

## Models

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| LSTM | 55-65% | Fast | Real-time |
| Transformer | 65-75% | Medium | Best accuracy |
| XGBoost | 60-70% | Very Fast | Ensemble |
| AutoGluon | 70-80% | Slow | Maximum accuracy |

## Features

Automatically generates 200+ features from OHLCV data:
- Bollinger Band metrics
- Momentum indicators (RSI, MACD)
- Volatility measures
- Volume analysis
- Price patterns
- Time-based features

## Performance

**Test Results (BTC 15m)**
- Accuracy: 72.3%
- AUC-ROC: 0.81
- Precision: 68.5%
- Recall: 75.2%

**Backtest (6 months)**
- Win Rate: 58.3%
- Sharpe Ratio: 1.85
- Total Return: 24.6%

## Documentation

See [docs/](docs/) for detailed guides:
- `SETUP.md` - Installation
- `USAGE.md` - How to use
- `MODEL_ARCHITECTURE.md` - Technical details
- `RESULTS.md` - Performance analysis

## License

MIT License
