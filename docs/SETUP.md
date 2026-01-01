# Setup & Installation Guide

## System Requirements

- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- GPU (CUDA 11.8+) for faster training (optional but recommended)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/caizongxun/bb-breakout-prediction.git
cd bb-breakout-prediction
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n bb-breakout python=3.9
conda activate bb-breakout
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```python
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__}');"
python -c "import pandas; print(f'Pandas {pandas.__version__}');"
python -c "import xgboost; print(f'XGBoost {xgboost.__version__}');"
```

## GPU Setup (Optional but Recommended)

### CUDA & cuDNN Setup

```bash
# Install CUDA 11.8 (https://developer.nvidia.com/cuda-11-8-0-download-archive)
# Install cuDNN 8.7+ (https://developer.nvidia.com/cudnn)

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### For M1/M2 Mac Users

```bash
# Use conda to install TensorFlow for Apple Silicon
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
```

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  symbols: ['BTCUSDT', 'ETHUSDT']  # Add your symbols
  timeframes: ['15m', '1h']

model:
  type: 'transformer'  # Choose: lstm, transformer, ensemble
  epochs: 50

features:
  n_features: 30
  auto_engineer: true
```

## Verify Everything Works

```bash
# Download sample data
python scripts/download_data.py --symbols BTCUSDT --timeframe 15m

# Train model (this will take 30 minutes)
python scripts/train_models.py --symbol BTCUSDT --model lstm --epochs 20

# Make predictions
python scripts/predict_realtime.py --symbol BTCUSDT --model best
```

## Troubleshooting

### Out of Memory Error

1. Reduce batch size in `config.yaml`:
   ```yaml
   model:
     batch_size: 16  # from 32
   ```

2. Reduce sequence length:
   ```yaml
   model:
     sequence_length: 20  # from 30
   ```

### TensorFlow Version Issues

```bash
# Uninstall and reinstall
pip uninstall tensorflow -y
pip install tensorflow==2.12.0
```

### CUDA Issues

```bash
# Disable GPU (slower but works)
export CUDA_VISIBLE_DEVICES=-1
python scripts/train_models.py ...
```

## Next Steps

See [USAGE.md](USAGE.md) for training and prediction workflows.
