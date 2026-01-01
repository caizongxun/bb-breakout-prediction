#!/usr/bin/env python3
"""
Main training script for BB Breakout Prediction models

Usage:
    python scripts/train_models.py --symbol BTCUSDT --model lstm --epochs 50
    python scripts/train_models.py --symbol ETHUSDT --model transformer
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.models import LSTMModel, TransformerModel, EnsembleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sequences(data, labels, seq_length=30):
    """Create sequence data for LSTM/Transformer"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels.iloc[i+seq_length] if hasattr(labels, 'iloc') else labels[i+seq_length])
    return np.array(X), np.array(y)


def train_lstm(X_train, X_val, X_test, y_train, y_val, y_test, epochs=50):
    """Train LSTM model"""
    logger.info("Training LSTM model...")
    
    model = LSTMModel.create(seq_length=X_train.shape[1], n_features=X_train.shape[2])
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    
    accuracy = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"LSTM Results - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    return model, history


def train_transformer(X_train, X_val, X_test, y_train, y_val, y_test, epochs=50):
    """Train Transformer model"""
    logger.info("Training Transformer model...")
    
    model = TransformerModel.create(seq_length=X_train.shape[1], n_features=X_train.shape[2])
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    
    accuracy = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Transformer Results - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train BB Breakout Prediction Models')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe (15m or 1h)')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer', 'ensemble'], help='Model type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='./data/models', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading {args.symbol} {args.timeframe} data...")
    loader = DataLoader()
    df = loader.load(args.symbol, args.timeframe)
    
    # Generate features
    logger.info("Generating features...")
    engineer = FeatureEngineer()
    df = engineer.generate_all_features(df)
    df = engineer.create_target(df)
    
    # Select features
    logger.info("Selecting features...")
    X = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'future_volatility', 'future_volatility_pct', 'target_breakout', 'target_magnitude']).dropna()
    y = df['target_breakout'].loc[X.index]
    
    selected_features, importance_df = engineer.select_features_by_importance(X, y, n_features=30)
    X_selected = X[selected_features]
    
    # Standardize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X_selected.index)
    
    # Create sequences
    seq_length = 30
    X_seq, y_seq = create_sequences(X_scaled.values, y.values, seq_length)
    
    # Split data (time series - no shuffle)
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.15)
    
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    
    X_val = X_seq[train_size:train_size+val_size]
    y_val = y_seq[train_size:train_size+val_size]
    
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Train model
    if args.model == 'lstm':
        import tensorflow as tf
        from tensorflow import keras
        model, history = train_lstm(X_train, X_val, X_test, y_train, y_val, y_test, args.epochs)
        model_path = output_dir / f"{args.symbol}_{args.model}.h5"
        model.save(model_path)
    
    elif args.model == 'transformer':
        import tensorflow as tf
        from tensorflow import keras
        model, history = train_transformer(X_train, X_val, X_test, y_train, y_val, y_test, args.epochs)
        model_path = output_dir / f"{args.symbol}_{args.model}.h5"
        model.save(model_path)
    
    # Save scaler and features
    scaler_path = output_dir / f"{args.symbol}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    features_path = output_dir / f"{args.symbol}_features.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(selected_features, f)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Features saved to {features_path}")


if __name__ == '__main__':
    main()
