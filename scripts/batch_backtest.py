#!/usr/bin/env python3
"""
Batch backtesting script for all trained models

Tests trading performance across all cryptocurrencies and timeframes.

Usage:
    python scripts/batch_backtest.py --model lstm --threshold 0.6
    python scripts/batch_backtest.py --model transformer --symbols BTCUSDT ETHUSDT
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
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for models"""
    
    def __init__(self, models_dir='./data/models', model_type='lstm'):
        self.models_dir = Path(models_dir)
        self.model_type = model_type
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        self.results = []
        
    def backtest_single(self, symbol, timeframe, entry_threshold=0.6, 
                       position_size=1.0, commission=0.001):
        """
        Backtest single symbol/timeframe
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
            
            # Load data
            df = self.loader.load(symbol, timeframe)
            df = self.engineer.generate_all_features(df)
            
            trades = []
            
            # Backtest
            for i in range(30, len(df) - 20):
                # Get recent window
                recent = df[selected_features].iloc[i-30:i].dropna()
                if len(recent) < 30:
                    continue
                
                recent_scaled = scaler.transform(recent)
                prob = model.predict(recent_scaled.reshape(1, 30, -1), verbose=0)[0][0]
                
                # Entry signal
                if prob > entry_threshold:
                    entry_price = df['close'].iloc[i]
                    entry_time = df.index[i]
                    
                    # Look ahead 20 candles
                    future_high = df['high'].iloc[i:i+20].max()
                    future_low = df['low'].iloc[i:i+20].min()
                    exit_time = df.index[min(i+20, len(df)-1)]
                    
                    # Simple exit: follow the trend
                    upside = future_high - entry_price
                    downside = entry_price - future_low
                    
                    if upside > downside:
                        exit_price = future_high
                        direction = 'LONG'
                    else:
                        exit_price = future_low
                        direction = 'SHORT'
                    
                    # Calculate P&L
                    if direction == 'LONG':
                        pnl = (exit_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - exit_price) / entry_price
                    
                    # Apply commission
                    pnl_after_comm = pnl - (2 * commission)  # Round trip
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': direction,
                        'probability': prob,
                        'pnl': pnl,
                        'pnl_after_comm': pnl_after_comm,
                        'profit': pnl_after_comm > 0
                    })
            
            if not trades:
                logger.warning(f"No trades for {symbol} {timeframe}")
                return None
            
            # Calculate statistics
            trades_df = pd.DataFrame(trades)
            
            total_trades = len(trades_df)
            winning_trades = (trades_df['pnl_after_comm'] > 0).sum()
            losing_trades = (trades_df['pnl_after_comm'] < 0).sum()
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = trades_df[trades_df['pnl_after_comm'] > 0]['pnl_after_comm'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl_after_comm'] < 0]['pnl_after_comm'].mean() if losing_trades > 0 else 0
            
            cumulative_returns = (1 + trades_df['pnl_after_comm']).cumprod() - 1
            total_return = cumulative_returns.iloc[-1]
            
            # Sharpe Ratio
            daily_returns = trades_df['pnl_after_comm'].values
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
            
            # Max Drawdown
            cumulative = (1 + trades_df['pnl_after_comm']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'total_trades': int(total_trades),
                'winning_trades': int(winning_trades),
                'losing_trades': int(losing_trades),
                'win_rate': float(win_rate),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown),
                'avg_probability': float(trades_df['probability'].mean()),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(
                f"✓ {symbol} {timeframe}: "
                f"Trades={total_trades}, "
                f"WinRate={win_rate:.2%}, "
                f"Return={total_return:.2%}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol} {timeframe}: {e}")
            return None
    
    def backtest_all(self, symbols=None, timeframes=None, entry_threshold=0.6,
                    num_workers=2):
        """
        Backtest all models
        """
        if symbols is None:
            model_files = list(self.models_dir.glob(f"*_{self.model_type}.h5"))
            symbols = sorted(set([f.stem.rsplit('_', 2)[0] for f in model_files]))
        
        if timeframes is None:
            timeframes = ['15m', '1h']
        
        logger.info(f"Backtesting {len(symbols)} symbols, {len(timeframes)} timeframes")
        
        # Backtest in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.backtest_single, symbol, timeframe, entry_threshold)
                for symbol in symbols
                for timeframe in timeframes
            ]
            
            for future in tqdm(futures, total=len(futures), desc="Backtesting"):
                result = future.result()
                if result:
                    self.results.append(result)
        
        return self.results
    
    def print_summary(self):
        """Print backtest summary"""
        if not self.results:
            logger.warning("No results")
            return
        
        results_df = pd.DataFrame(self.results)
        
        logger.info("\n" + "="*100)
        logger.info("BACKTEST SUMMARY")
        logger.info("="*100)
        
        logger.info(f"Total Models: {len(results_df)}")
        logger.info(f"Total Trades: {results_df['total_trades'].sum()}")
        logger.info(f"Overall Win Rate: {(results_df['winning_trades'].sum() / results_df['total_trades'].sum()):.2%}")
        logger.info(f"Overall Return: {results_df['total_return'].mean():.2%}")
        logger.info(f"Mean Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f}")
        logger.info(f"Mean Max Drawdown: {results_df['max_drawdown'].mean():.2%}")
        
        # By timeframe
        logger.info("\nBy Timeframe:")
        for tf in sorted(results_df['timeframe'].unique()):
            tf_data = results_df[results_df['timeframe'] == tf]
            logger.info(
                f"  {tf}: "
                f"Models={len(tf_data)}, "
                f"AvgReturn={tf_data['total_return'].mean():.2%}, "
                f"AvgSharpe={tf_data['sharpe_ratio'].mean():.2f}"
            )
        
        # Top performers
        logger.info("\nTop 10 by Return:")
        top_return = results_df.nlargest(10, 'total_return')
        for _, row in top_return.iterrows():
            logger.info(
                f"  {row['symbol']:12} {row['timeframe']:3} | "
                f"Return: {row['total_return']:7.2%} | "
                f"Sharpe: {row['sharpe_ratio']:6.2f} | "
                f"WinRate: {row['win_rate']:6.2%}"
            )
        
        logger.info("\n" + "="*100 + "\n")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description='Batch Backtest All Models')
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Model type')
    parser.add_argument('--models-dir', type=str, default='./data/models',
                       help='Models directory')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Entry probability threshold')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Specific symbols')
    parser.add_argument('--timeframes', nargs='+', default=None,
                       help='Specific timeframes')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of parallel workers')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Backtest
    engine = BacktestEngine(
        models_dir=args.models_dir,
        model_type=args.model
    )
    
    results = engine.backtest_all(
        symbols=args.symbols,
        timeframes=args.timeframes,
        entry_threshold=args.threshold,
        num_workers=args.workers
    )
    
    # Print summary
    results_df = engine.print_summary()
    
    # Save if requested
    if args.output and not results_df.empty:
        results_df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
