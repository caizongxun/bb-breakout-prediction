# 批量訓練指南

本文檔說明如何使用批量訓練系統訓練所有幣種的模型。

## 快速開始

### 1. 訓練所有幣種 (LSTM - 快速)

```bash
# 使用2個並行工作進程，每個模型訓練50個epochs
python scripts/batch_train_all.py \
    --model lstm \
    --epochs 50 \
    --workers 2
```

### 2. 訓練所有幣種 (Transformer - 高精度)

```bash
# Transformer模型需要更多時間但精度更高
python scripts/batch_train_all.py \
    --model transformer \
    --epochs 100 \
    --workers 2
```

### 3. 訓練特定幣種

```bash
# 只訓練BTC和ETH
python scripts/batch_train_all.py \
    --model lstm \
    --symbols BTCUSDT ETHUSDT \
    --workers 1
```

### 4. 訓練特定時間框架

```bash
# 只訓練15分鐘K線
python scripts/batch_train_all.py \
    --model lstm \
    --timeframes 15m \
    --workers 2
```

### 5. 從中斷處恢復訓練

```bash
# 如果訓練中途中斷，可以恢復並繼續
python scripts/batch_train_all.py \
    --model lstm \
    --workers 2 \
    --resume
```

## 支持的幣種 (20種)

```
AAVEUSDT, ADAUSDT, ALGOUSDT, ARBUSDT, ATOMUSDT,
AVAXUSDT, BCHUSDT, BNBUSDT, BICUSDT, DOGEUSD,
DOTUSDT, ETCUSDT, ETHUSDT, FILUSDT, LINKUSDT,
LTCUSDT, MATICUSDT, NEARUSDT, OPUSDT, SOLUSDT,
UNIUSDT, XRPUSDT
```

## 時間框架

- `15m` - 15分鐘K線
- `1h` - 1小時K線

## 訓練時間估計

| 配置 | 模型 | 幣種數 | 時間框架 | 總訓練時間 | 並行工作進程 |
|------|------|--------|---------|----------|----------|
| 快速 | LSTM | 5 | 2 | ~20分鐘 | 2 |
| 中等 | LSTM | 20 | 2 | ~2-3小時 | 2 |
| 高質量 | Transformer | 20 | 2 | ~6-8小時 | 2 |
| 完整 | 兩個都 | 20 | 2 | ~10-12小時 | 2 |

**注意**: 時間取決於硬件配置和數據大小

## 並行工作進程

```
--workers N
```

推薦配置:
- **GPU**: 2-4個工作進程 (不要過多導致顯存爆炸)
- **CPU**: 1-2個工作進程

## 輸出文件

訓練完成後，在 `./data/models/` 目錄下會生成:

```
data/models/
├── training_results.json          # 訓練結果摘要
├── BTCUSDT_15m_lstm.h5           # 模型文件
├── BTCUSDT_15m_scaler.pkl        # 特徵縮放器
├── BTCUSDT_15m_features.pkl      # 選中的特徵列表
├── ETHUSDT_15m_lstm.h5
├── ETHUSDT_15m_scaler.pkl
├── ETHUSDT_15m_features.pkl
└── ...
```

## 監控訓練進度

### 1. 實時日誌

```bash
tail -f logs/batch_training.log
```

### 2. 查看結果

```bash
python -c \
  "import json; \
   results = json.load(open('./data/models/training_results.json')); \
   print(f'已訓練: {len(results)} 個模型')"
```

## 訓練結果分析

### Python API

```python
import json
import pandas as pd

# 加載結果
with open('./data/models/training_results.json') as f:
    results = json.load(f)

# 轉換為DataFrame
results_list = list(results.values())
df = pd.DataFrame(results_list)

# 按時間框架分析
print("\n按15m時間框架分析:")
df_15m = df[df['timeframe'] == '15m']
print(f"平均精度: {df_15m['accuracy'].mean():.4f}")
print(f"平均AUC: {df_15m['auc'].mean():.4f}")

# 找出表現最好的模型
top_models = df.nlargest(5, 'auc')
print("\nTop 5 模型:")
print(top_models[['symbol', 'timeframe', 'accuracy', 'auc']])
```

## 常見問題

### Q: 訓練中途斷電怎麼辦?
A: 使用 `--resume` 標籤恢復訓練。已完成的模型會被跳過，只訓練未完成的。

### Q: 顯存不足?
A: 
1. 減少 `--workers` 數量
2. 修改 `config.yaml` 中的 batch_size (32 → 16)
3. 減少 sequence_length (30 → 20)

### Q: 訓練速度太慢?
A:
1. 增加 `--workers` 數量
2. 使用GPU (檢查 `CUDA_VISIBLE_DEVICES`)
3. 減少 `--epochs` 數量

### Q: 某個幣種訓練失敗?
A: 檢查日誌:
```bash
grep "BTCUSDT" logs/batch_training.log | grep "Error"
```
常見原因:
- 數據不足
- NaN值過多
- 特徵生成失敗

## 下一步: 批量預測

訓練完成後，使用訓練好的模型進行批量預測:

```bash
# 對所有模型進行預測
python scripts/batch_predict.py \
    --model lstm \
    --confidence-threshold 0.6 \
    --probability-threshold 0.6 \
    --output predictions.csv
```

## 下一步: 批量回測

評估模型的交易性能:

```bash
# 對所有模型進行回測
python scripts/batch_backtest.py \
    --model lstm \
    --threshold 0.6 \
    --workers 2 \
    --output backtest_results.csv
```

## 批量訓練工作流程

```
1. 批量訓練所有幣種
   python scripts/batch_train_all.py --model lstm --workers 2
   ↓
2. 檢查訓練結果
   查看 training_results.json
   ↓
3. 批量預測 (實時)
   python scripts/batch_predict.py --model lstm
   ↓
4. 批量回測 (評估績效)
   python scripts/batch_backtest.py --model lstm
   ↓
5. 分析結果
   找出表現最好的模型組合
   ↓
6. 開始交易
   使用表現最好的模型進行實盤
```

## 性能基準

基於20個幣種、2個時間框架 (40個模型):

### LSTM (40個模型)
- 訓練時間: ~2-3小時
- 平均精度: 58-62%
- 平均AUC: 0.62-0.68

### Transformer (40個模型)
- 訓練時間: ~6-8小時
- 平均精度: 65-72%
- 平均AUC: 0.70-0.78

## 最佳實踐

1. **從LSTM開始** - 快速驗證數據和管道
2. **檢查結果** - 分析 training_results.json
3. **調整參數** - 根據結果優化config.yaml
4. **訓練Transformer** - 得到更高精度
5. **組合預測** - 使用LSTM+Transformer集成
6. **定期重訓練** - 每月使用新數據重訓

## 故障排除

### 查看完整錯誤信息

```bash
python scripts/batch_train_all.py --model lstm --workers 1
```

### 檢查特定幣種

```python
import sys
sys.path.insert(0, '.')
from src.data_loader import DataLoader

loader = DataLoader()
df = loader.load('BTCUSDT', '15m')
print(f"數據量: {len(df)}")
print(f"時間範圍: {df.index.min()} 到 {df.index.max()}")
```

### 測試模型加載

```python
import tensorflow as tf
model = tf.keras.models.load_model('./data/models/BTCUSDT_15m_lstm.h5')
print(model.summary())
```
