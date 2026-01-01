# 快速開始指南

## 安裝 (2分鐘)

```bash
# 1. 克隆倉庫
git clone https://github.com/caizongxun/bb-breakout-prediction.git
cd bb-breakout-prediction

# 2. 創建虛擬環境
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. 安裝依賴
pip install -r requirements.txt
```

## 訓練單個模型 (30分鐘)

```bash
# 訓練BTC 15分鐘K線 - LSTM模型
python scripts/train_models.py --symbol BTCUSDT --model lstm --epochs 30

# 結果會保存到 data/models/ 目錄
```

## 訓練所有幣種 (3-8小時)

### 快速方式 (LSTM - 更快)
```bash
python scripts/batch_train_all.py --model lstm --epochs 50 --workers 2
```

### 高精度方式 (Transformer - 更準確)
```bash
python scripts/batch_train_all.py --model transformer --epochs 50 --workers 2
```

### 監控訓練進度
```bash
# 在新終端查看日誌
tail -f logs/batch_training.log
```

## 訓練完成後

### 查看結果
```bash
# Windows PowerShell
Get-Content data/models/training_results.json | ConvertFrom-Json | ConvertTo-Json

# Mac/Linux
cat data/models/training_results.json | python -m json.tool
```

### 實時預測 (檢查當前爆發信號)
```bash
python scripts/batch_predict.py --model lstm --output alerts.csv
```

### 驗證策略性能
```bash
python scripts/batch_backtest.py --model lstm --output backtest_results.csv
```

## 遇到問題?

### 問題1: "logs目錄不存在"
- 已修復，自動創建
- 或手動創建: `mkdir logs`

### 問題2: GPU內存不足
```bash
# 減少並行工作進程
python scripts/batch_train_all.py --model lstm --workers 1 --epochs 30
```

### 問題3: 訓練太慢
```bash
# 增加並行工作進程 (需要足夠GPU)
python scripts/batch_train_all.py --model lstm --workers 4

# 或減少epochs
python scripts/batch_train_all.py --model lstm --epochs 30
```

### 問題4: 訓練中斷後恢復
```bash
# 使用 --resume 參數
python scripts/batch_train_all.py --model lstm --workers 2 --resume
```

## 文件結構

訓練完成後，你會看到:

```
data/models/
├── training_results.json       # 所有訓練結果
├── BTCUSDT_15m_lstm.h5         # BTC 15m LSTM模型
├── BTCUSDT_15m_scaler.pkl      # BTC 15m 特徵縮放器
├── BTCUSDT_15m_features.pkl    # BTC 15m 特徵列表
├── ETHUSDT_15m_lstm.h5
├── ETHUSDT_15m_scaler.pkl
├── ETHUSDT_15m_features.pkl
└── ... (共40個模型)

logs/
└── batch_training.log          # 訓練日誌
```

## 預期時間

| 操作 | 時間 | 說明 |
|------|------|------|
| 安裝 | 2分 | 首次安裝較慢 |
| 訓練1個模型 | 5分 | LSTM單個模型 |
| 訓練5個模型 | 20分 | 快速測試 |
| 訓練40個模型 | 2-3小時 | LSTM全套 |
| 訓練40個模型 | 6-8小時 | Transformer全套 |
| 預測 | 5分 | 所有40個模型 |
| 回測 | 1小時 | 所有40個模型 |

## 下一步

1. **查看文檔**: 詳細信息見 [docs/BATCH_TRAINING.md](docs/BATCH_TRAINING.md)
2. **配置調整**: 編輯 `config.yaml` 自定義參數
3. **性能優化**: 根據GPU能力調整 `--workers` 數量
4. **實盤交易**: 使用訓練好的模型進行實時預測

## 常用命令

```bash
# 訓練 (LSTM - 推薦)
python scripts/batch_train_all.py --model lstm --epochs 50 --workers 2

# 監控
tail -f logs/batch_training.log

# 預測
python scripts/batch_predict.py --model lstm

# 回測
python scripts/batch_backtest.py --model lstm

# 只訓練特定幣種
python scripts/batch_train_all.py --symbols BTCUSDT ETHUSDT --model lstm

# 恢復訓練
python scripts/batch_train_all.py --model lstm --resume
```

## 成功標誌

✓ 訓練完成，無錯誤  
✓ data/models/ 目錄下有訓練好的模型  
✓ training_results.json 包含40個結果  
✓ logs/batch_training.log 記錄完整  
✓ 模型平均AUC > 0.60  

## 需要幫助?

查看完整文檔: [docs/](docs/)  
GitHub Issues: [提交問題](https://github.com/caizongxun/bb-breakout-prediction/issues)  

---

**快樂交易!** 🚀
