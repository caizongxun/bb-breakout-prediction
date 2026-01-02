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
import warnings
import os
import json

import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class SimplePredictor(keras.layers.Layer):
    """简单的预测层 - 用於兼容旧模型"""
    def __init__(self, **kwargs):
        super(SimplePredictor, self).__init__(**kwargs)
    
    def call(self, inputs):
        return inputs


class ModelLoader:
    """加載和使用訓練好的模型"""
    
    def __init__(self, model_dir='./data/models', converted_model_dir='./data/models_converted'):
        """
        初始化模型加載器
        
        Args:
            model_dir: 原始模型目录
            converted_model_dir: 转换后的模型目录
        """
        self.model_dir = Path(model_dir)
        self.converted_model_dir = Path(converted_model_dir)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    def list_available_models(self):
        """
        列出所有可用的訓練模型
        
        Returns:
            dict: {symbol: {timeframe: [model_types]}}
        """
        models = {}
        
        # 扫描所有 .h5 文件（修復：不用 | 操作符）
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
        獲取模型文件路径
        
        Args:
            symbol: 币种符号 (e.g., 'BTCUSDT')
            timeframe: 时间框架 (e.g., '15m', '1h')
            model_type: 模型类型 ('transformer' 或 'lstm')
        
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
    
    def get_converted_model_path(self, symbol, timeframe, model_type='transformer'):
        """
        獲取转换后的模型路径
        
        Returns:
            str or None
        """
        converted_path = self.converted_model_dir / f"{symbol}_{timeframe}_{model_type}"
        if converted_path.exists():
            return str(converted_path)
        return None
    
    def load_model(self, symbol, timeframe, model_type='transformer'):
        """
        加載訓練好的模型及其配置
        
        优先级:
        1. 转换后的 SavedModel 格式
        2. 原始 H5 格式 (不同的加载方法)
        3. 虛擬模型
        
        Args:
            symbol: 币种符号
            timeframe: 时间框架
            model_type: 模型类型
        
        Returns:
            tuple: (model, scaler, features_list)
        """
        logger.info(f"Loading model: {symbol} {timeframe} ({model_type})")
        
        # 优先级 1: 检查转换后的模型
        converted_path = self.get_converted_model_path(symbol, timeframe, model_type)
        if converted_path:
            logger.info(f"Found converted model at: {converted_path}")
            try:
                logger.info("Loading converted SavedModel...")
                model = keras.saving.load_model(converted_path)
                logger.info("✓ Converted model loaded successfully")
                
                # 丢弃 H5 模型，正常加载 scaler 和 features
                paths = self.get_model_path(symbol, timeframe, model_type)
                with open(paths['scaler'], 'rb') as f:
                    scaler = pickle.load(f)
                with open(paths['features'], 'rb') as f:
                    features = pickle.load(f)
                
                logger.info(f"Successfully loaded: {len(features)} features")
                return model, scaler, features
            except Exception as e:
                logger.warning(f"Failed to load converted model: {e}")
        
        # 优先级 2-4: 加載原始 H5 模型
        paths = self.get_model_path(symbol, timeframe, model_type)
        logger.info(f"Model path: {paths['model']}")
        
        model = None
        
        # 方法 1: 标准加載
        try:
            logger.info("Trying standard load method...")
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model = keras.models.load_model(paths['model'])
            logger.info("✓ Model loaded successfully with standard method")
        except Exception as e1:
            logger.warning(f"Standard load failed: {e1}")
            
            # 方法 2: 自定义对象
            try:
                logger.info("Trying custom objects method...")
                custom_objects = {'SimplePredictor': SimplePredictor}
                model = keras.models.load_model(
                    paths['model'],
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy', keras.metrics.AUC()]
                )
                logger.info("✓ Model loaded with custom objects")
            except Exception as e2:
                logger.warning(f"Custom objects method failed: {e2}")
                
                # 方法 3: H5 重建
                try:
                    logger.info("Trying H5 reconstruction method...")
                    model = self._reconstruct_from_h5(paths['model'])
                    if model:
                        logger.info("✓ Model reconstructed from H5")
                    else:
                        raise Exception("H5 reconstruction returned None")
                except Exception as e3:
                    logger.warning(f"H5 reconstruction failed: {e3}")
                    
                    # 方法 4: 虛擬模型（僅用於测试）
                    logger.warning("All loading methods failed. Using dummy model for testing.")
                    logger.warning("WARNING: 这只是用於测试的虛擬模型，预测结果不会准確！")
                    model = self._create_dummy_model()
        
        # 加載 scaler
        with open(paths['scaler'], 'rb') as f:
            scaler = pickle.load(f)
        
        # 加載特徵列表
        with open(paths['features'], 'rb') as f:
            features = pickle.load(f)
        
        logger.info(f"Successfully loaded: {len(features)} features")
        
        return model, scaler, features
    
    def _reconstruct_from_h5(self, h5_path):
        """
        从 H5 文件重建模型
        """
        try:
            import h5py
            logger.info("Attempting to reconstruct model from H5 weights...")
            
            with h5py.File(h5_path, 'r') as f:
                logger.info(f"H5 structure: {list(f.keys())}")
                
                if 'model_config' in f.attrs:
                    config_str = f.attrs['model_config']
                    if isinstance(config_str, bytes):
                        config_str = config_str.decode('utf-8')
                    config = json.loads(config_str)
                    logger.info(f"Model config found")
                    
                    try:
                        model = keras.Sequential.from_config(config)
                        return model
                    except:
                        pass
            
            return None
        except Exception as e:
            logger.error(f"H5 reconstruction failed: {e}")
            return None
    
    def _create_dummy_model(self, input_shape=(30, 30)):
        """
        创建虛擬模型用于测试
        """
        logger.warning("Creating dummy model for testing...")
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def prepare_data(self, df, features, scaler, seq_length=30):
        """
        準备数据用于预测
        
        Args:
            df: 包含所有特征的 DataFrame (已經过特征工程)
            features: 要使用的特征列表
            scaler: MinMaxScaler 对象
            seq_length: 序列長度
        
        Returns:
            np.array: 形状为 (1, seq_length, n_features) 的序列
        """
        # 只使用指定的特征
        X = df[features].tail(seq_length).values
        
        if len(X) < seq_length:
            logger.warning(f"Not enough data: got {len(X)}, need {seq_length}")
            return None
        
        # 标准化
        X_scaled = scaler.transform(X)
        
        # 转换为模型输入格式
        X_seq = np.array([X_scaled])
        
        return X_seq
    
    def predict(self, df, model, scaler, features, seq_length=30):
        """
        使用模型进行预测
        
        Args:
            df: 特征 DataFrame
            model: 訓練好的模型
            scaler: Scaler 对象
            features: 特征列表
            seq_length: 序列長度
        
        Returns:
            dict with prediction results
        """
        # 準备数据
        X_seq = self.prepare_data(df, features, scaler, seq_length)
        
        if X_seq is None:
            return None
        
        # 预测（禁止警告）
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            y_prob = model.predict(X_seq, verbose=0)[0][0]
        
        y_pred = 1 if y_prob > 0.5 else 0
        
        return {
            'prediction': int(y_pred),
            'probability': float(y_prob),
            'confidence': float(max(y_prob, 1 - y_prob)),
            'signal': '看涨 (Breakout)' if y_pred == 1 else '看跌 (No Breakout)',
            'strength': float(y_prob) if y_pred == 1 else float(1 - y_prob)
        }
    
    def batch_predict(self, df, symbol, timeframe, model_type='transformer'):
        """
        使用指定模型批量预测
        
        Args:
            df: 特征 DataFrame
            symbol: 币种符号
            timeframe: 时间框架
            model_type: 模型类型
        
        Returns:
            dict with prediction
        """
        model, scaler, features = self.load_model(symbol, timeframe, model_type)
        result = self.predict(df, model, scaler, features)
        return result


class ModelSelector:
    """帮助选择最佳模型的工具"""
    
    def __init__(self, results_file='./data/models/training_results.json'):
        """
        初始化模型选择器
        
        Args:
            results_file: 訓練结果 JSON 文件
        """
        self.results_file = Path(results_file)
        self.results = self._load_results()
    
    def _load_results(self):
        """
        加載訓練结果
        """
        if not self.results_file.exists():
            return {}
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_best_model(self, symbol, timeframe):
        """
        獲取指定符号和时间框架的最佳模型
        
        Returns:
            dict with model info and performance
        """
        key = f"{symbol}_{timeframe}"
        if key in self.results:
            return self.results[key]
        return None
    
    def get_best_for_symbol(self, symbol):
        """
        獲取某个符号的最佳时间框架
        
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
        推荐表现最好的前 N 个模型
        
        Returns:
            list of best models sorted by AUC
        """
        results_list = list(self.results.values())
        results_list.sort(key=lambda x: x.get('auc', 0), reverse=True)
        return results_list[:top_n]
    
    def get_model_stats(self, symbol=None):
        """
        獲取模型统计信息
        
        Args:
            symbol: 特定符号，或 None 表示全部
        
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
