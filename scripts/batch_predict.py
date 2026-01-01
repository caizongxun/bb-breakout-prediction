#!/usr/bin/env python3
"""
Batch prediction script for all trained models

Generates predictions for all cryptocurrencies across timeframes,
useful for monitoring multiple coins in real-time.

Usage:
    python scripts/batch_predict.py --model lstm --confidence 0.6
    python scripts/batch_predict.py --model transformer --output alerts.csv
"""

import argparse
import logging
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchPredictor:
    """Batch prediction manager"""
    
    def __init__(self, models_dir='./data/models', model_type='lstm'):
        self.models_dir = Path(models_dir)
        self.model_type = model_type
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        self.predictions = []
        
    def predict_single(self, symbol, timeframe):
        """
        Make prediction for single symbol/timeframe
        """
        try:
            import tensorflow as tf
            
            # Load model
            model_path = self.models_dir / f"{symbol}_{timeframe}_{self.model_type}.h5"
            if not model_path.exists():
                logger.warning(f"Model not found: {model_path}")
                return None
            
            model = tf.keras.models.load_model(model_path)
            
            # Load scaler and features
            scaler_path = self.models_dir / f"{symbol}_{timeframe}_scaler.pkl"
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            features_path = self.models_dir / f"{symbol}_{timeframe}_features.pkl"
            with open(features_path, 'rb') as f:
                selected_features = pickle.load(f)
            
            # Load latest data
            df = self.loader.load(symbol, timeframe)
            df = self.engineer.generate_all_features(df)
            
            # Prepare recent data
            recent = df[selected_features].iloc[-30:].dropna()
            if len(recent) < 30:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return None
            
            # Scale
            recent_scaled = scaler.transform(recent)
            
            # Predict
            prob = model.predict(recent_scaled.reshape(1, 30, -1), verbose=0)[0][0]
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'probability': float(prob),
                'alert': 'YES' if prob > 0.6 else 'NO',
                'confidence': float(abs(prob - 0.5) * 2),
                'latest_price': float(df['close'].iloc[-1]),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting {symbol} {timeframe}: {e}")
            return None
    
    def predict_all(self, symbols=None, timeframes=None):
        """
        Make predictions for all models
        """
        if symbols is None:
            # Get symbols from existing models
            model_files = list(self.models_dir.glob(f"*_{self.model_type}.h5"))
            symbols = sorted(set([f.stem.rsplit('_', 2)[0] for f in model_files]))
        
        if timeframes is None:
            timeframes = ['15m', '1h']
        
        logger.info(f"Making predictions for {len(symbols)} symbols, {len(timeframes)} timeframes")
        
        results = []
        for symbol in symbols:
            for timeframe in timeframes:
                result = self.predict_single(symbol, timeframe)
                if result:
                    results.append(result)
        
        self.predictions = results
        return results
    
    def to_dataframe(self):
        """Convert predictions to DataFrame"""
        return pd.DataFrame(self.predictions)
    
    def save_csv(self, output_path):
        """Save predictions to CSV"""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        return df
    
    def print_alerts(self, confidence_threshold=0.6, probability_threshold=0.6):
        """Print high-confidence alerts"""
        df = self.to_dataframe()
        
        # Filter alerts
        alerts = df[
            (df['confidence'] >= confidence_threshold) & 
            (df['probability'] >= probability_threshold)
        ].sort_values('probability', ascending=False)
        
        if alerts.empty:
            logger.info("No alerts at current thresholds")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"BREAKOUT ALERTS ({len(alerts)} found)")
        logger.info(f"{'='*80}")
        
        for _, row in alerts.iterrows():
            logger.info(
                f"[{row['symbol']} {row['timeframe']}] "
                f"Probability: {row['probability']:.2%}, "
                f"Confidence: {row['confidence']:.2%}, "
                f"Price: {row['latest_price']:.2f}"
            )
        
        logger.info(f"{'='*80}\n")
        
        return alerts
    
    def get_top_predictions(self, n=10):
        """Get top N predictions by probability"""
        df = self.to_dataframe()
        return df.nlargest(n, 'probability')
    
    def summary_by_timeframe(self):
        """Print summary by timeframe"""
        df = self.to_dataframe()
        
        logger.info("\nSummary by Timeframe:")
        for tf in sorted(df['timeframe'].unique()):
            tf_data = df[df['timeframe'] == tf]
            avg_prob = tf_data['probability'].mean()
            n_alerts = (tf_data['alert'] == 'YES').sum()
            
            logger.info(
                f"  {tf}: {len(tf_data)} models, "
                f"Avg Prob={avg_prob:.2%}, "
                f"Alerts={n_alerts}"
            )


def main():
    parser = argparse.ArgumentParser(description='Batch Predict for All Models')
    parser.add_argument('--model', type=str, default='lstm', 
                       choices=['lstm', 'transformer'],
                       help='Model type')
    parser.add_argument('--models-dir', type=str, default='./data/models',
                       help='Models directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file')
    parser.add_argument('--confidence-threshold', type=float, default=0.6,
                       help='Confidence threshold for alerts')
    parser.add_argument('--probability-threshold', type=float, default=0.6,
                       help='Probability threshold for alerts')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Specific symbols to predict')
    parser.add_argument('--timeframes', nargs='+', default=None,
                       help='Specific timeframes')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = BatchPredictor(
        models_dir=args.models_dir,
        model_type=args.model
    )
    
    # Make predictions
    predictions = predictor.predict_all(
        symbols=args.symbols,
        timeframes=args.timeframes
    )
    
    logger.info(f"\nTotal Predictions: {len(predictions)}")
    
    # Print alerts
    predictor.print_alerts(
        confidence_threshold=args.confidence_threshold,
        probability_threshold=args.probability_threshold
    )
    
    # Print summary
    predictor.summary_by_timeframe()
    
    # Print top predictions
    logger.info("\nTop 10 Predictions:")
    top_df = predictor.get_top_predictions(10)
    for _, row in top_df.iterrows():
        logger.info(
            f"  {row['symbol']:12} {row['timeframe']:3} | "
            f"Prob: {row['probability']:6.2%} | "
            f"Conf: {row['confidence']:6.2%}"
        )
    
    # Save if requested
    if args.output:
        predictor.save_csv(args.output)


if __name__ == '__main__':
    main()
