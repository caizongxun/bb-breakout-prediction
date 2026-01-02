#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安裝棄棂棂棂棂棂棂 - 棂棂棂棂 Flask 前作䬛
"""

import os
import sys
from pathlib import Path

def check_directory():
    """棂棂棂棂目录結構"""
    print("\n[1] 棂棂棂棂節目錄")
    print("=" * 60)
    
    required_dirs = [
        'src',
        'templates',
        'data/models',
        'scripts',
        'logs'
    ]
    
    for d in required_dirs:
        path = Path(d)
        status = "✓" if path.exists() else "✗"
        print(f"{status} {d:<20} {'OK' if path.exists() else 'NOT FOUND'}")
    
    return all(Path(d).exists() for d in required_dirs[:3])  # 核寶前三個

def check_files():
    """棂棂階档案案"""
    print("\n[2] 棂棂階档案案")
    print("=" * 60)
    
    required_files = {
        'app.py': '主Flask應用',
        'src/model_loader.py': '模型加載器',
        'src/data_loader.py': '數據加載器',
        'src/feature_engineering.py': '特徵工程',
        'templates/index.html': '主頁面HTML',
        'templates/error.html': '錯誤頁面HTML',
    }
    
    all_exist = True
    for file, desc in required_files.items():
        path = Path(file)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {file:<30} {desc}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_models():
    """棂棂模式model"""
    print("\n[3] 棂棂模式模型")
    print("=" * 60)
    
    models_dir = Path('./data/models')
    if not models_dir.exists():
        print("✗ models目錄不存在")
        return False
    
    h5_files = list(models_dir.glob('*.h5'))
    if not h5_files:
        print("✗ 找model檔案")
        return False
    
    print(f"✓ 找到 {len(h5_files)} 個模式模型\n")
    
    # 汲劲前 5 個
    for h5_file in h5_files[:5]:
        symbol_info = h5_file.stem
        print(f"  • {h5_file.name}")
    
    if len(h5_files) > 5:
        print(f"  ... 及其他 {len(h5_files) - 5} 個")
    
    return len(h5_files) > 0

def check_dependencies():
    """棂棂依賴套件"""
    print("\n[4] 棂棂依賴套件")
    print("=" * 60)
    
    required_packages = [
        'flask',
        'plotly',
        'pandas',
        'numpy',
        'tensorflow',
        'scikit-learn',
        'xgboost',
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package:<20} OK")
        except ImportError:
            print(f"✗ {package:<20} NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_model_loader():
    """棂棂 ModelLoader 亟系統"""
    print("\n[5] 棂棂 ModelLoader 系統")
    print("=" * 60)
    
    try:
        from src.model_loader import ModelLoader, ModelSelector
        
        ml = ModelLoader()
        available = ml.list_available_models()
        
        print(f"✓ ModelLoader 載入成功")
        print(f"✓ 找到 {len(available)} 種交易對")
        
        for symbol, timeframes in list(available.items())[:3]:
            print(f"  • {symbol}: {', '.join(timeframes.keys())}")
        
        return len(available) > 0
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        return False

def check_flask():
    """棂棂 Flask 應用亟系統"""
    print("\n[6] 棂棂 Flask 應用")
    print("=" * 60)
    
    try:
        from app import app, model_loader, model_selector
        
        print("✓ Flask 應用載入成功")
        print("✓ 模型配置已載入")
        
        # 檢查路由
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        print(f"✓ 找到 {len(routes)} 個 API 路由")
        
        for route in sorted(routes):
            if route.startswith('/api') or route == '/':
                print(f"  • {route}")
        
        return True
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("🔨 Flask Web App 安裝棄棂檢查")
    print("="*60)
    
    results = {
        '目錄結構': check_directory(),
        '需要案案': check_files(),
        '模式model': check_models(),
        '依賴套件': check_dependencies(),
        'ModelLoader': check_model_loader(),
        'Flask應用': check_flask(),
    }
    
    print("\n" + "="*60)
    print("➡️  檢查結果統計")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ 通過" if result else "✗ 失敗"
        print(f"{status:<10} {name}")
    
    print(f"\n檢查結果: {passed}/{total}")
    
    if passed == total:
        print("\n🌟 汲劲前全部正常！")
        print("\n您可以执行:")
        print("  python app.py")
        print("\n然後打開: http://localhost:5000")
        return 0
    else:
        print("\n⚠️  有些項目需要修正，請棄棂上叶的錯誤提示。")
        return 1

if __name__ == '__main__':
    sys.exit(main())
