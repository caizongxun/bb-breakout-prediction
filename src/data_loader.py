#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loader for Cryptocurrency OHLCV Data
Supports loading from HuggingFace with automatic timestamp parsing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os

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
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import huggingface_hub
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            raise
        
        try:
            # Extract base symbol (remove USDT)
            base_symbol = symbol.replace('USDT', '').replace('USD', '')
            
            url = f"https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/resolve/main/klines/{symbol}/{base_symbol}_{timeframe}.parquet"
            
            file_path = self.cache_dir / f"{symbol}_{timeframe}.parquet"
            
            # Download if not exists
            if not file_path.exists():
                logger.info(f"Downloading {symbol} {timeframe} data from HuggingFace...")
                try:
                    df = pd.read_parquet(url)
                    df.to_parquet(file_path)
                    logger.info(f"Successfully downloaded and cached {len(df)} candles")
                except Exception as e:
                    logger.warning(f"Failed to download from remote: {e}")
                    raise
            else:
                logger.info(f"Loading {symbol} {timeframe} from cache...")
                df = pd.read_parquet(file_path)
                logger.info(f"Loaded {len(df)} candles from cache")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in download_from_huggingface for {symbol} {timeframe}: {e}")
            raise
    
    def _parse_timestamp(self, timestamp_col):
        """
        Parse timestamp column with automatic format detection
        
        Handles:
        - Integer milliseconds (e.g., 1234567890000)
        - ISO datetime strings (e.g., '2023-04-26T12:00:00')
        - Datetime strings with space (e.g., '2023-04-26 12:00:00')
        - Already datetime objects
        
        Returns:
            DatetimeIndex
        """
        # If already datetime, return as is
        if pd.api.types.is_datetime64_any_dtype(timestamp_col):
            return timestamp_col
        
        # Try to infer and parse
        try:
            # First try direct conversion (handles most cases)
            return pd.to_datetime(timestamp_col, infer_datetime_format=True)
        except Exception as e1:
            logger.debug(f"Direct conversion failed: {e1}")
            
            # If column is object/string type
            if timestamp_col.dtype == 'object':
                first_val = str(timestamp_col.iloc[0]) if len(timestamp_col) > 0 else ''
                
                # Check if it looks like integer milliseconds
                if first_val.isdigit() and len(first_val) >= 10:
                    try:
                        logger.info("Parsing timestamp as millisecond integers")
                        return pd.to_datetime(pd.to_numeric(timestamp_col), unit='ms')
                    except Exception as e2:
                        logger.warning(f"Millisecond parsing failed: {e2}")
            
            # If numeric dtype, assume milliseconds
            if pd.api.types.is_numeric_dtype(timestamp_col):
                try:
                    logger.info("Parsing timestamp as numeric milliseconds")
                    return pd.to_datetime(timestamp_col, unit='ms')
                except Exception as e3:
                    logger.warning(f"Numeric millisecond parsing failed: {e3}")
            
            # Last resort: try with different format strings
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                try:
                    logger.info(f"Trying format: {fmt}")
                    return pd.to_datetime(timestamp_col, format=fmt)
                except Exception:
                    continue
            
            logger.error(f"Could not parse timestamp. First value: {first_val}")
            raise ValueError(f"Unable to parse timestamp column. Sample: {first_val}")
    
    def preprocess(self, df):
        """
        Clean and prepare data
        
        Args:
            df: Raw dataframe from source
        
        Returns:
            Cleaned dataframe with proper index and dtypes
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        logger.info(f"Preprocessing {len(df)} candles...")
        
        # Rename columns to standard format if needed
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Time': 'timestamp',
            'Timestamp': 'timestamp',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
        
        # Parse timestamp with smart detection
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = self._parse_timestamp(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                logger.info("Timestamp parsed and set as index")
            except Exception as e:
                logger.error(f"Error parsing timestamp: {e}")
                raise
        elif df.index.name and 'time' in df.index.name.lower():
            # If index is already timestamp-like
            try:
                df.index = self._parse_timestamp(df.index)
                logger.info("Index parsed as timestamp")
            except Exception as e:
                logger.warning(f"Could not parse index as timestamp: {e}")
        
        # Remove NaN
        original_len = len(df)
        df = df.dropna()
        logger.info(f"Removed {original_len - len(df)} rows with NaN")
        
        # Remove duplicates (keep first occurrence)
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by time (index should be datetime)
        try:
            df = df.sort_index()
            logger.info(f"Data sorted by timestamp")
        except Exception as e:
            logger.warning(f"Could not sort by index: {e}")
        
        logger.info(f"Preprocessing complete: {len(df)} candles remaining")
        return df
    
    def load(self, symbol, timeframe='15m'):
        """
        Load and preprocess cryptocurrency data
        
        Args:
            symbol: e.g., 'BTCUSDT', 'ETHUSDT'
            timeframe: '15m' or '1h'
        
        Returns:
            Preprocessed DataFrame with timestamp as index
        """
        logger.info(f"Loading {symbol} {timeframe}...")
        df = self.download_from_huggingface(symbol, timeframe)
        df = self.preprocess(df)
        return df
