#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
訓練好的模型加載和推理模塊

一個模型有 3 個檔案：
  1. .h5 - 神經網路模型權重
  2. _scaler.pkl - 特徵標準化器
  3. _features.pkl - 選中的特徵列表

使用方式:
  loader = ModelLoader()
  model, scaler, features = loader.load_model('BTCUSDT', '15m', 'transformer')
  prediction = loader.predict(df, model, scaler, features)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging

try:
    from tensorflow import keras
except ImportError:
    keras = None

logger = logging.getLogger(__name__)


class ModelLoader:
    """加載和使用訓練好的模型"""
    
    def __init__(self, model_dir='./data/models'):
        """
        初始化模型加載器
        
        Args:
            model_dir: 模型文件目錄
        """
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    def list_available_models(self):
        """
        列出所有可用的訓練模型
        
        Returns:
            dict: {symbol: {timeframe: [model_types]}}
        """
        models = {}
        
        # 掃描所有 .h5 文件（修復：不用 | 操作符）
        h5_files = list(self.model_dir.glob('*_transformer.h5')) + list(self.model_dir.glob('*_lstm.h5'))
        
        for h5_file in h5_files:
            # 解析文件名
            parts = h5_file.stem.split('_')
            if len(parts) >= 3:
                symbol = parts[0]
                timeframe = parts[1]
                model_type = parts[2]
                
                if symbol not in models:
                    models[symbol] = {}
                if timeframe not in models[symbol]:
                    models[symbol][timeframe] = []
                
                if model_type not in models[symbol][timeframe]:
                    models[symbol][timeframe].append(model_type)
        
        return models
    
    def get_model_path(self, symbol, timeframe, model_type='transformer'):
        """
        獲取模型文件路徑
        
        Args:
            symbol: 幣種符號 (e.g., 'BTCUSDT')
            timeframe: 時間框架 (e.g., '15m', '1h')
            model_type: 模型類型 ('transformer' 或 'lstm')
        
        Returns:
            dict with paths to model, scaler, features
        """
        model_file = self.model_dir / f"{symbol}_{timeframe}_{model_type}.h5"
        scaler_file = self.model_dir / f"{symbol}_{timeframe}_scaler.pkl"
        features_file = self.model_dir / f"{symbol}_{timeframe}_features.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_file}")
        if not features_file.exists():
            raise FileNotFoundError(f"Features not found: {features_file}")
        
        return {
            'model': str(model_file),
            'scaler': str(scaler_file),
            'features': str(features_file)
        }
    
    def load_model(self, symbol, timeframe, model_type='transformer'):
        """
        加載訓練好的模型及其配置
        
        Args:
            symbol: 幣種符號
            timeframe: 時間框架
            model_type: 模型類型
        
        Returns:
            tuple: (model, scaler, features_list)
        """
        if keras is None:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        paths = self.get_model_path(symbol, timeframe, model_type)
        
        # 加載模型
        logger.info(f"Loading model: {symbol} {timeframe} ({model_type})")
        model = keras.models.load_model(paths['model'])
        
        # 加載 scaler
        with open(paths['scaler'], 'rb') as f:
            scaler = pickle.load(f)
        
        # 加載特徵列表
        with open(paths['features'], 'rb') as f:
            features = pickle.load(f)
        
        logger.info(f"Successfully loaded: {len(features)} features")
        
        return model, scaler, features
    
    def prepare_data(self, df, features, scaler, seq_length=30):
        """
        準備數據用於預測
        
        Args:
            df: 包含所有特徵的 DataFrame (已經過特徵工程)
            features: 要使用的特徵列表
            scaler: MinMaxScaler 對象
            seq_length: 序列長度
        
        Returns:
            np.array: 形狀為 (1, seq_length, n_features) 的序列
        """
        # 只使用指定的特徵
        X = df[features].tail(seq_length).values
        
        if len(X) < seq_length:
            logger.warning(f"Not enough data: got {len(X)}, need {seq_length}")
            return None
        
        # 標準化
        X_scaled = scaler.transform(X)
        
        # 轉換為模型輸入格式
        X_seq = np.array([X_scaled])
        
        return X_seq
    
    def predict(self, df, model, scaler, features, seq_length=30):
        """
        使用模型進行預測
        
        Args:
            df: 特徵 DataFrame
            model: 訓練好的模型
            scaler: Scaler 對象
            features: 特徵列表
            seq_length: 序列長度
        
        Returns:
            dict with prediction results
        """
        # 準備數據
        X_seq = self.prepare_data(df, features, scaler, seq_length)
        
        if X_seq is None:
            return None
        
        # 預測
        y_prob = model.predict(X_seq, verbose=0)[0][0]
        y_pred = 1 if y_prob > 0.5 else 0
        
        return {
            'prediction': int(y_pred),
            'probability': float(y_prob),
            'confidence': float(max(y_prob, 1 - y_prob)),
            'signal': '看漲 (Breakout)' if y_pred == 1 else '看跌 (No Breakout)',
            'strength': float(y_prob) if y_pred == 1 else float(1 - y_prob)
        }
    
    def batch_predict(self, df, symbol, timeframe, model_type='transformer'):
        """
        使用指定模型批量預測
        
        Args:
            df: 特徵 DataFrame
            symbol: 幣種符號
            timeframe: 時間框架
            model_type: 模型類型
        
        Returns:
            dict with prediction
        """
        model, scaler, features = self.load_model(symbol, timeframe, model_type)
        result = self.predict(df, model, scaler, features)
        return result


class ModelSelector:
    """幫助選擇最佳模型的工具"""
    
    def __init__(self, results_file='./data/models/training_results.json'):
        """
        初始化模型選擇器
        
        Args:
            results_file: 訓練結果 JSON 文件
        """
        self.results_file = Path(results_file)
        self.results = self._load_results()
    
    def _load_results(self):
        """
        加載訓練結果
        """
        if not self.results_file.exists():
            return {}
        
        import json
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_best_model(self, symbol, timeframe):
        """
        獲取指定符號和時間框架的最佳模型
        
        Returns:
            dict with model info and performance
        """
        key = f"{symbol}_{timeframe}"
        if key in self.results:
            return self.results[key]
        return None
    
    def get_best_for_symbol(self, symbol):
        """
        獲取某個符號的最佳時間框架
        
        Returns:
            tuple: (best_timeframe, best_model_info)
        """
        best_auc = -1
        best_timeframe = None
        best_result = None
        
        for key, result in self.results.items():
            if result.get('symbol') == symbol:
                if result.get('auc', 0) > best_auc:
                    best_auc = result['auc']
                    best_timeframe = result['timeframe']
                    best_result = result
        
        return best_timeframe, best_result
    
    def recommend_models(self, top_n=10):
        """
        推薦表現最好的前 N 個模型
        
        Returns:
            list of best models sorted by AUC
        """
        results_list = list(self.results.values())
        results_list.sort(key=lambda x: x.get('auc', 0), reverse=True)
        return results_list[:top_n]
    
    def get_model_stats(self, symbol=None):
        """
        獲取模型統計信息
        
        Args:
            symbol: 特定符號，或 None 表示全部
        
        Returns:
            dict with statistics
        """
        results_list = list(self.results.values())
        
        if symbol:
            results_list = [r for r in results_list if r.get('symbol') == symbol]
        
        if not results_list:
            return None
        
        accuracies = [r['accuracy'] for r in results_list]
        aucs = [r['auc'] for r in results_list]
        
        return {
            'count': len(results_list),
            'mean_accuracy': float(np.mean(accuracies)),
            'mean_auc': float(np.mean(aucs)),
            'best_accuracy': float(np.max(accuracies)),
            'best_auc': float(np.max(aucs)),
            'worst_accuracy': float(np.min(accuracies)),
            'worst_auc': float(np.min(aucs))
        }
