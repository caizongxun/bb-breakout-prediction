#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BB 通道預測可視化 Web 應用

功能:
  - 加載訓練好的模型
  - 預測 BB 通道突破
  - 实时顯示預測結果和可信度
  - 可視化技術指標和 BB 通道

運行:
  pip install flask plotly pandas numpy tensorflow scikit-learn
  python app.py
  
  打開瀏覽器: http://localhost:5000
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from src.data_loader import DataLoader
    from src.feature_engineering import FeatureEngineer
    from src.model_loader import ModelLoader, ModelSelector
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the project directory and have installed dependencies.")

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 創建 Flask 應用
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False

# 全局變数
data_loader = DataLoader()
feature_engineer = FeatureEngineer()
model_loader = ModelLoader()
model_selector = ModelSelector()


@app.route('/')
def index():
    """
    主頁 - 模型選擇和配置
    """
    try:
        # 獲取可用模型
        available_models = model_loader.list_available_models()
        
        # 獲取推薦模型
        recommended = model_selector.recommend_models(top_n=5)
        
        return render_template('index.html',
                             available_models=available_models,
                             recommended_models=recommended)
    except Exception as e:
        logger.error(f"Error in index: {e}")
        return render_template('error.html', error=str(e)), 500


@app.route('/api/models')
def get_models():
    """
    API: 獲取所有可用模型列表
    """
    try:
        available_models = model_loader.list_available_models()
        
        # 轉換格式
        models_list = []
        for symbol, timeframes in available_models.items():
            for timeframe, model_types in timeframes.items():
                for model_type in model_types:
                    # 獲取模型性能
                    result = model_selector.get_best_model(symbol, timeframe)
                    models_list.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'model_type': model_type,
                        'accuracy': result.get('accuracy', 0) if result else 0,
                        'auc': result.get('auc', 0) if result else 0,
                        'f1': result.get('f1', 0) if result else 0
                    })
        
        return jsonify({
            'status': 'success',
            'data': models_list,
            'count': len(models_list)
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API: 進行預測
    
    請求參數:
      - symbol: 幣種符號 (e.g., 'BTCUSDT')
      - timeframe: 時間框架 (e.g., '15m', '1h')
      - model_type: 模型類型 ('transformer' 或 'lstm')
    
    返回: 預測結果和 BB 通道圖表
    """
    try:
        data = request.json
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        model_type = data.get('model_type', 'transformer')
        
        if not symbol or not timeframe:
            return jsonify({'status': 'error', 'message': '缺少必要參数'}), 400
        
        logger.info(f"Predicting for {symbol} {timeframe}")
        
        # 加載數據
        df = data_loader.load(symbol, timeframe)
        logger.info(f"Loaded {len(df)} candles")
        
        # 特徵工程
        df = feature_engineer.generate_all_features(df)
        df = feature_engineer.create_target(df, forward_window=5)
        
        # 加載模型並預測
        model, scaler, features = model_loader.load_model(symbol, timeframe, model_type)
        prediction = model_loader.predict(df, model, scaler, features)
        
        if prediction is None:
            return jsonify({'status': 'error', 'message': '數據不足'}), 400
        
        # 生成圖表
        chart_data = generate_chart(df, symbol, timeframe, prediction)
        
        # 準備統計數據
        stats = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_candles': len(df),
            'last_price': float(df['close'].iloc[-1]),
            'last_time': str(df.index[-1]),
            'bb_upper': float(df['bb_upper_20'].iloc[-1]) if 'bb_upper_20' in df.columns else None,
            'bb_middle': float(df['close'].iloc[-1]),  # 使用当前价格作为BB中线的佋子
            'bb_lower': float(df['bb_lower_20'].iloc[-1]) if 'bb_lower_20' in df.columns else None,
        }
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'stats': stats,
            'chart': chart_data
        })
    
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        return jsonify({'status': 'error', 'message': f'模型不存在: {e}'}), 404
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/chart', methods=['POST'])
def get_chart():
    """
    API: 獲取 K 線圖表 (支持參數調整)
    
    請求參数:
      - symbol: 幣種符號
      - timeframe: 時間框架
      - days: 顯示最近 N 天的數據
    """
    try:
        data = request.json
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        days = data.get('days', 30)
        
        # 加載數據
        df = data_loader.load(symbol, timeframe)
        
        # 特徵工程
        df = feature_engineer.generate_all_features(df)
        
        # 只顯示最後 N 天
        if days > 0:
            cutoff_date = df.index[-1] - timedelta(days=days)
            df = df[df.index >= cutoff_date]
        
        # 生成圖表
        chart_data = generate_chart(df, symbol, timeframe)
        
        return jsonify({
            'status': 'success',
            'chart': chart_data
        })
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def generate_chart(df, symbol, timeframe, prediction=None):
    """
    生成 K 線圖表，包含 BB 通道和技術指標
    
    Args:
        df: 包含 OHLCV 和技術指標的 DataFrame
        symbol: 幣種符號
        timeframe: 時間框架
        prediction: 預測結果（可選）
    
    Returns:
        Plotly JSON 数据
    """
    # 确保指標存在
    if 'bb_upper_20' not in df.columns:
        df = feature_engineer.generate_all_features(df)
    
    # 建立子圖
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # K 線图
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # BB 中线 (20日標准剧)
    bb_middle_col = 'bb_upper_20'  # 使用存在的列
    if 'bb_upper_20' in df.columns:
        # 计算中线 (SMA 20)
        bb_middle = df['close'].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=bb_middle,
                name='BB 中线',
                line=dict(color='blue', width=1),
                mode='lines'
            ),
            row=1, col=1
        )
        
        # BB 上軌
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_upper_20'],
                name='BB 上軌',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                mode='lines',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # BB 下軌
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_lower_20'],
                name='BB 下軌',
                line=dict(color='rgba(0, 255, 0, 0.3)', width=1),
                mode='lines',
                fill='tonexty',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    # RSI
    if 'rsi_14' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi_14'],
                name='RSI',
                line=dict(color='orange', width=1),
                mode='lines'
            ),
            row=2, col=1
        )
        
        # RSI 过买/过卖线
        fig.add_hline(y=70, line_dash='dash', line_color='red', 
                     annotation_text='过买', row=2, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color='green',
                     annotation_text='过卖', row=2, col=1)
    
    # 如果有预渫结果，添加标题
    title = f"{symbol} {timeframe} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if prediction:
        signal = prediction.get('signal', '')
        prob = prediction.get('probability', 0)
        title += f" | {signal} (概率: {prob:.2%})"
    
    # 更新图表布局
    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="价格",
        height=600,
        template='plotly_white',
        hovermode='x unified',
        font=dict(size=10)
    )
    
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    # 转换为 JSON
    return plotly.io.to_json(fig)


@app.route('/api/stats')
def get_stats():
    """
    API: 獲取模型統計信息
    """
    try:
        symbol = request.args.get('symbol')
        
        stats = model_selector.get_model_stats(symbol=symbol)
        
        if not stats:
            return jsonify({'status': 'error', 'message': '没有統計数据'}), 404
        
        return jsonify({
            'status': 'success',
            'data': stats
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/recommended')
def get_recommended():
    """
    API: 獲取推薦的最佳模型
    """
    try:
        top_n = request.args.get('top_n', 10, type=int)
        
        recommended = model_selector.recommend_models(top_n=top_n)
        
        return jsonify({
            'status': 'success',
            'data': recommended,
            'count': len(recommended)
        })
    except Exception as e:
        logger.error(f"Recommended error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='页面不存在'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='服勊器错误'), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("BB 通道預測可視化 Web 应用")
    print("="*80)
    print("\n正在启动服务器...")
    print("打开浏覽器: http://localhost:5000\n")
    
    # 启动 Flask 应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )
