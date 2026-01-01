import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage cryptocurrency OHLCV data from HuggingFace"""
    
    def __init__(self, cache_dir='./data/raw'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_from_huggingface(self, symbol, timeframe='15m'):
        """
        Download data from HuggingFace dataset
        
        Args:
            symbol: e.g., 'BTCUSDT', 'ETHUSDT'
            timeframe: '15m' or '1h'
        """
        try:
            import huggingface_hub
            
            url = f"https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/resolve/main/klines/{symbol}/{symbol.split('USDT')[0]}_{timeframe}.parquet"
            
            file_path = self.cache_dir / f"{symbol}_{timeframe}.parquet"
            
            # Download if not exists
            if not file_path.exists():
                logger.info(f"Downloading {symbol} {timeframe} data...")
                df = pd.read_parquet(url)
                df.to_parquet(file_path)
                logger.info(f"Downloaded {len(df)} candles to {file_path}")
            else:
                df = pd.read_parquet(file_path)
                logger.info(f"Loaded {len(df)} candles from cache")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def preprocess(self, df):
        """
        Clean and prepare data
        """
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        
        # Remove NaN
        df = df.dropna()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by time
        df = df.sort_index()
        
        logger.info(f"Preprocessed: {len(df)} candles")
        return df
    
    def load(self, symbol, timeframe='15m'):
        """
        Load and preprocess data
        """
        df = self.download_from_huggingface(symbol, timeframe)
        df = self.preprocess(df)
        return df
