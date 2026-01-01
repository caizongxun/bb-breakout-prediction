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
    
    def _parse_timestamp(self, timestamp_col):
        """
        Parse timestamp column with automatic format detection
        Handles: millisecond integers, datetime strings, etc.
        """
        if timestamp_col.dtype == 'object':
            # Try to detect format from first value
            first_val = str(timestamp_col.iloc[0])
            
            # Check if it looks like a datetime string
            if isinstance(first_val, str) and '-' in first_val:
                # Format like '2023-04-26 12:00:00'
                try:
                    return pd.to_datetime(timestamp_col)
                except Exception:
                    pass
            
            # Try parsing as number (milliseconds)
            try:
                return pd.to_datetime(pd.to_numeric(timestamp_col), unit='ms')
            except Exception:
                pass
        
        # If numeric, treat as milliseconds
        if pd.api.types.is_numeric_dtype(timestamp_col):
            try:
                return pd.to_datetime(timestamp_col, unit='ms')
            except Exception:
                pass
        
        # Default: assume it's already datetime or try direct conversion
        try:
            return pd.to_datetime(timestamp_col)
        except Exception as e:
            logger.warning(f"Could not parse timestamp: {e}")
            return timestamp_col
    
    def preprocess(self, df):
        """
        Clean and prepare data
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Convert timestamp with smart detection
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = self._parse_timestamp(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                logger.info(f"Timestamp parsed successfully")
            except Exception as e:
                logger.error(f"Error parsing timestamp: {e}")
                raise
        
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
