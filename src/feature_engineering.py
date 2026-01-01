import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Automatic feature generation and selection"""
    
    @staticmethod
    def generate_all_features(df):
        """
        Generate 200+ features from OHLCV data
        """
        df = df.copy()
        
        # Price features
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_above_sma_{period}'] = (df['close'] > df[f'sma_{period}']).astype(int)
        
        # Volatility
        for period in [10, 20, 50]:
            df[f'volatility_{period}'] = df['return'].rolling(period).std()
        
        # Volume
        df['vol_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['vol_change'] = df['volume'].pct_change()
        
        # Momentum
        for period in [5, 10, 14, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # RSI variants
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for period in [10, 20, 30]:
            bb_std = 2
            bb_sma = df['close'].rolling(period).mean()
            bb_std_val = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = bb_sma + bb_std * bb_std_val
            df[f'bb_lower_{period}'] = bb_sma - bb_std * bb_std_val
            df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_width_{period}'] + 1e-8)
        
        # MACD variants
        for fast, slow, signal in [(5, 13, 5), (12, 26, 9), (8, 17, 9)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}_{signal}'] = macd.ewm(span=signal).mean()
            df[f'macd_hist_{fast}_{slow}_{signal}'] = macd - macd.ewm(span=signal).mean()
        
        # Time features
        df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
        df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
        
        logger.info(f"Generated {df.shape[1] - 5} features")
        return df
    
    @staticmethod
    def select_features_by_importance(X, y, n_features=30, cumulative_importance=0.85):
        """
        Select features using XGBoost feature importance
        Filters out non-numeric columns automatically
        """
        # Filter to only numeric columns
        X_numeric = X.select_dtypes(include=[np.number]).copy()
        
        if X_numeric.shape[1] == 0:
            logger.warning("No numeric features found. Returning first n_features as fallback.")
            return X.columns[:n_features].tolist(), pd.DataFrame()
        
        logger.info(f"Using {X_numeric.shape[1]} numeric features for importance selection")
        
        # Train XGBoost on numeric features only
        try:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_numeric, y)
            
            importance_df = pd.DataFrame({
                'feature': X_numeric.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select by cumulative importance or n_features
            cumsum = importance_df['importance'].cumsum() / importance_df['importance'].sum()
            selected = importance_df[cumsum <= cumulative_importance]['feature'].tolist()
            
            # Ensure minimum features
            if len(selected) < n_features:
                selected = importance_df.head(n_features)['feature'].tolist()
            
            logger.info(f"Selected {len(selected)} features (importance: {cumulative_importance:.2f})")
            return selected, importance_df
            
        except Exception as e:
            logger.warning(f"XGBoost feature selection failed: {e}. Using top numeric features.")
            # Fallback: return top numeric features by variance
            variances = X_numeric.var().sort_values(ascending=False)
            return variances.head(n_features).index.tolist(), pd.DataFrame()
    
    @staticmethod
    def create_target(df, forward_window=5, volatility_threshold_pct=1.5):
        """
        Create target variable for breakout prediction
        """
        df = df.copy()
        
        # Future volatility
        df['future_volatility'] = df['high'].rolling(forward_window).max() - df['low'].rolling(forward_window).min()
        df['future_volatility_pct'] = (df['future_volatility'] / df['close']) * 100
        
        # Binary target: breakout or not
        df['target_breakout'] = (df['future_volatility_pct'].shift(-forward_window) > volatility_threshold_pct).astype(int)
        
        # Magnitude target
        df['target_magnitude'] = df['future_volatility_pct'].shift(-forward_window)
        
        logger.info(f"Created target. Breakout rate: {df['target_breakout'].mean():.2%}")
        return df
