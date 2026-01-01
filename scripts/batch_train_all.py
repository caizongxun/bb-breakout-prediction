#!/usr/bin/env python3
"""
Batch training script for all cryptocurrencies across multiple timeframes

This script trains models for all supported symbols and timeframes in parallel,
with automatic checkpointing and progress tracking.

Usage:
    python scripts/batch_train_all.py --model transformer --epochs 50 --workers 4
    python scripts/batch_train_all.py --model lstm --workers 2 --resume
"""

import argparse
import logging
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import sys
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.models import LSTMModel, TransformerModel

# Create logs directory
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'batch_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# All supported cryptocurrencies
SYMBOLS = [
    'AAVEUDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
    'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BICUSDT', 'DOGEUSD',
    'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LINKUSDT',
    'LTCUSDT', 'MATICUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT',
    'UNIUSDT', 'XRPUSDT'
]

TIMEFRAMES = ['15m', '1h']


class BatchTrainer:
    """Batch training manager for multiple symbols and timeframes"""
    
    def __init__(self, model_type='lstm', epochs=50, output_dir='./data/models'):
        self.model_type = model_type
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.results = {}
        self.results_file = self.output_dir / 'training_results.json'
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        
    def create_sequences(self, data, labels, seq_length=30):
        """Create sequence data for neural networks"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(labels.iloc[i+seq_length] if hasattr(labels, 'iloc') else labels[i+seq_length])
        return np.array(X), np.array(y)
    
    def train_single_model(self, symbol, timeframe):
        """
        Train a single model for a symbol and timeframe
        """
        try:
            logger.info(f"Training {symbol} {timeframe} ({self.model_type})...")
            
            # Load data
            df = self.loader.load(symbol, timeframe)
            if len(df) < 500:  # Minimum data requirement
                logger.warning(f"Skipping {symbol} {timeframe}: insufficient data ({len(df)} candles)")
                return None
            
            # Feature engineering
            df = self.engineer.generate_all_features(df)
            df = self.engineer.create_target(df, forward_window=5)
            
            # Prepare data
            feature_cols = [col for col in df.columns if col not in 
                          ['open', 'high', 'low', 'close', 'volume', 'future_volatility', 
                           'future_volatility_pct', 'target_breakout', 'target_magnitude']]
            
            X = df[feature_cols].dropna()
            y = df['target_breakout'].loc[X.index]
            
            if len(X) < 100:  # Minimum after NaN removal
                logger.warning(f"Skipping {symbol} {timeframe}: insufficient data after cleaning")
                return None
            
            # Feature selection
            selected_features, _ = self.engineer.select_features_by_importance(
                X, y, n_features=30
            )
            X_selected = X[selected_features]
            
            # Standardize
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_selected)
            X_scaled = pd.DataFrame(X_scaled, columns=selected_features, index=X_selected.index)
            
            # Create sequences
            seq_length = 30
            X_seq, y_seq = self.create_sequences(X_scaled.values, y.values, seq_length)
            
            if len(X_seq) < 100:  # Minimum sequences
                logger.warning(f"Skipping {symbol} {timeframe}: insufficient sequences")
                return None
            
            # Split data (time series - no shuffle)
            train_size = int(len(X_seq) * 0.7)
            val_size = int(len(X_seq) * 0.15)
            
            X_train = X_seq[:train_size]
            y_train = y_seq[:train_size]
            X_val = X_seq[train_size:train_size+val_size]
            y_val = y_seq[train_size:train_size+val_size]
            X_test = X_seq[train_size+val_size:]
            y_test = y_seq[train_size+val_size:]
            
            # Train model
            import tensorflow as tf
            from tensorflow import keras
            
            if self.model_type == 'lstm':
                model = LSTMModel.create(seq_length, X_train.shape[2], lstm_units=128)
            else:  # transformer
                model = TransformerModel.create(seq_length, X_train.shape[2])
            
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0  # Suppress output for batch processing
            )
            
            # Evaluate
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            precision = (y_pred[y_test == 1].sum() / (y_pred.sum() + 1e-8))
            recall = (y_test.sum() > 0) and (y_pred[y_test == 1].sum() / (y_test.sum() + 1e-8)) or 0
            
            # Save model
            model_path = self.output_dir / f"{symbol}_{timeframe}_{self.model_type}.h5"
            model.save(model_path)
            
            # Save scaler and features
            scaler_path = self.output_dir / f"{symbol}_{timeframe}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            features_path = self.output_dir / f"{symbol}_{timeframe}_features.pkl"
            with open(features_path, 'wb') as f:
                pickle.dump(selected_features, f)
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'model_type': self.model_type,
                'accuracy': float(accuracy),
                'auc': float(auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(2 * precision * recall / (precision + recall + 1e-8)),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': len(selected_features),
                'epochs_trained': len(history.history['loss']),
                'model_path': str(model_path),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✓ {symbol} {timeframe}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"✗ Error training {symbol} {timeframe}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def train_all(self, symbols=None, timeframes=None, num_workers=2, resume=False):
        """
        Train models for all symbols and timeframes
        
        Args:
            symbols: List of symbols to train (default: all)
            timeframes: List of timeframes (default: ['15m', '1h'])
            num_workers: Number of parallel workers
            resume: Resume from last checkpoint
        """
        symbols = symbols or SYMBOLS
        timeframes = timeframes or TIMEFRAMES
        
        # Load previous results if resuming
        completed = set()
        if resume and self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
                completed = {(r['symbol'], r['timeframe']) for r in self.results.values()}
            logger.info(f"Resuming from {len(completed)} completed models")
        
        # Create task list
        tasks = [(s, t) for s in symbols for t in timeframes 
                if (s, t) not in completed]
        
        logger.info(f"Training {len(tasks)} models ({self.model_type}) with {num_workers} workers")
        logger.info(f"Symbols: {len(symbols)}, Timeframes: {len(timeframes)}")
        logger.info(f"Epochs: {self.epochs}")
        
        # Train in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.train_single_model, symbol, timeframe)
                for symbol, timeframe in tasks
            ]
            
            for i, future in enumerate(tqdm(futures, total=len(futures), desc="Training Progress")):
                result = future.result()
                if result:
                    key = f"{result['symbol']}_{result['timeframe']}"
                    self.results[key] = result
                    
                    # Save results incrementally
                    if (i + 1) % 5 == 0:
                        self.save_results()
        
        # Final save
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to JSON"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved: {len(self.results)} models")
    
    def print_summary(self):
        """Print training summary statistics"""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        results_list = list(self.results.values())
        
        logger.info("\n" + "="*80)
        logger.info("BATCH TRAINING SUMMARY")
        logger.info("="*80)
        
        logger.info(f"Total Models Trained: {len(results_list)}")
        logger.info(f"Model Type: {self.model_type}")
        
        # Group by timeframe
        by_tf = {}
        for r in results_list:
            tf = r['timeframe']
            if tf not in by_tf:
                by_tf[tf] = []
            by_tf[tf].append(r)
        
        for tf in sorted(by_tf.keys()):
            results_tf = by_tf[tf]
            accuracies = [r['accuracy'] for r in results_tf]
            aucs = [r['auc'] for r in results_tf]
            
            logger.info(f"\nTimeframe: {tf}")
            logger.info(f"  Models: {len(results_tf)}")
            logger.info(f"  Avg Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            logger.info(f"  Avg AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
            logger.info(f"  Best Accuracy: {np.max(accuracies):.4f} ({results_tf[np.argmax(accuracies)]['symbol']})")
            logger.info(f"  Best AUC: {np.max(aucs):.4f} ({results_tf[np.argmax(aucs)]['symbol']})")
        
        # Overall statistics
        all_accuracies = [r['accuracy'] for r in results_list]
        all_aucs = [r['auc'] for r in results_list]
        all_f1s = [r['f1'] for r in results_list]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Overall Performance:")
        logger.info(f"  Mean Accuracy: {np.mean(all_accuracies):.4f}")
        logger.info(f"  Mean AUC: {np.mean(all_aucs):.4f}")
        logger.info(f"  Mean F1: {np.mean(all_f1s):.4f}")
        logger.info(f"  Min Accuracy: {np.min(all_accuracies):.4f}")
        logger.info(f"  Max Accuracy: {np.max(all_accuracies):.4f}")
        logger.info(f"{'='*80}\n")
        
        # Top performers
        sorted_by_auc = sorted(results_list, key=lambda x: x['auc'], reverse=True)
        logger.info("Top 5 Models by AUC:")
        for i, r in enumerate(sorted_by_auc[:5], 1):
            logger.info(f"  {i}. {r['symbol']} {r['timeframe']}: {r['auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Batch Train Models for All Cryptocurrencies')
    parser.add_argument('--model', type=str, default='lstm', 
                       choices=['lstm', 'transformer'], 
                       help='Model type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to train (default: all)')
    parser.add_argument('--timeframes', nargs='+', default=None, help='Timeframes (default: 15m 1h)')
    parser.add_argument('--output-dir', type=str, default='./data/models', help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BatchTrainer(
        model_type=args.model,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    # Train
    trainer.train_all(
        symbols=args.symbols,
        timeframes=args.timeframes,
        num_workers=args.workers,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
