#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
View training results and identify models that need retraining

Usage:
    python scripts/view_results.py
    python scripts/view_results.py --min-auc 0.60
    python scripts/view_results.py --export csv
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime


def load_results(results_file='./data/models/training_results.json'):
    """Load training results from JSON file"""
    path = Path(results_file)
    if not path.exists():
        print(f"結果文件不存在: {results_file}")
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_summary(results_dict):
    """Print summary of training results"""
    if not results_dict:
        print("沒有訓練結果")
        return
    
    results_list = list(results_dict.values())
    df = pd.DataFrame(results_list)
    
    print("\n" + "="*100)
    print("訓練結果摘要")
    print("="*100)
    
    print(f"\n總訓練模型數: {len(results_list)}")
    print(f"成功訓練: {len(results_list)}")
    print(f"失敗: 0\n")
    
    # 按時間框架分組
    for tf in sorted(df['timeframe'].unique()):
        tf_data = df[df['timeframe'] == tf]
        print(f"時間框架: {tf}")
        print(f"  模型數: {len(tf_data)}")
        print(f"  平均準確度: {tf_data['accuracy'].mean():.4f} (+/- {tf_data['accuracy'].std():.4f})")
        print(f"  平均 AUC: {tf_data['auc'].mean():.4f} (+/- {tf_data['auc'].std():.4f})")
        print(f"  最高準確度: {tf_data['accuracy'].max():.4f} ({tf_data.loc[tf_data['accuracy'].idxmax(), 'symbol']})")
        print(f"  最低準確度: {tf_data['accuracy'].min():.4f} ({tf_data.loc[tf_data['accuracy'].idxmin(), 'symbol']})")
        print()
    
    # 整體統計
    print("="*100)
    print("整體性能")
    print("="*100)
    print(f"平均準確度: {df['accuracy'].mean():.4f}")
    print(f"平均 AUC: {df['auc'].mean():.4f}")
    print(f"平均 F1: {df['f1'].mean():.4f}")
    print(f"最高準確度: {df['accuracy'].max():.4f}")
    print(f"最低準確度: {df['accuracy'].min():.4f}")
    print()


def print_detailed_results(results_dict, sort_by='auc', reverse=True):
    """Print detailed results for each model"""
    if not results_dict:
        return
    
    results_list = list(results_dict.values())
    df = pd.DataFrame(results_list)
    
    # Sort by specified column
    df = df.sort_values(by=sort_by, ascending=not reverse)
    
    print("\n" + "="*120)
    print(f"詳細結果 (按 {sort_by} 排序)")
    print("="*120)
    print(f"{'Symbol':<12} {'TF':<6} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*120)
    
    for idx, row in df.iterrows():
        print(f"{row['symbol']:<12} {row['timeframe']:<6} {row['accuracy']:.4f}     {row['auc']:.4f}     "
              f"{row['precision']:.4f}     {row['recall']:.4f}     {row['f1']:.4f}")
    print()


def print_needs_retraining(results_dict, min_auc=0.60, max_accuracy=0.65):
    """Print models that need retraining (low AUC or accuracy)"""
    if not results_dict:
        return
    
    results_list = list(results_dict.values())
    df = pd.DataFrame(results_list)
    
    # Find models that need retraining
    needs_retrain = df[(df['auc'] < min_auc) | (df['accuracy'] < max_accuracy)]
    
    if len(needs_retrain) == 0:
        print(f"\n✓ 所有模型性能良好 (AUC >= {min_auc}, Accuracy >= {max_accuracy})")
        return
    
    print("\n" + "="*100)
    print(f"需要重新訓練的模型 (AUC < {min_auc} 或 Accuracy < {max_accuracy})")
    print("="*100)
    print(f"總數: {len(needs_retrain)}\n")
    print(f"{'Symbol':<12} {'TF':<6} {'Accuracy':<10} {'AUC':<10} {'原因':<30}")
    print("-"*100)
    
    for idx, row in needs_retrain.iterrows():
        reasons = []
        if row['auc'] < min_auc:
            reasons.append(f"AUC={row['auc']:.4f}<{min_auc}")
        if row['accuracy'] < max_accuracy:
            reasons.append(f"Accuracy={row['accuracy']:.4f}<{max_accuracy}")
        reason = ", ".join(reasons)
        print(f"{row['symbol']:<12} {row['timeframe']:<6} {row['accuracy']:.4f}     {row['auc']:.4f}     {reason:<30}")
    
    # Generate command to retrain
    print("\n重新訓練命令:")
    retrain_symbols = needs_retrain['symbol'].unique()
    cmd = f"python scripts/batch_train_all.py --symbols {' '.join(retrain_symbols)} --epochs 50 --workers 2"
    print(cmd)
    print()


def export_results(results_dict, format='csv', output_file=None):
    """Export results to CSV or Excel"""
    if not results_dict:
        return
    
    results_list = list(results_dict.values())
    df = pd.DataFrame(results_list)
    
    # Remove model_path column for export
    export_cols = ['symbol', 'timeframe', 'model_type', 'accuracy', 'auc', 'precision', 'recall', 'f1', 'train_size', 'test_size']
    df_export = df[export_cols]
    
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'training_results_{timestamp}.{format}'
    
    if format == 'csv':
        df_export.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"已導出到: {output_file}")
    elif format == 'excel':
        df_export.to_excel(output_file, index=False)
        print(f"已導出到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='View training results')
    parser.add_argument('--min-auc', type=float, default=0.60, help='最小 AUC 閾值')
    parser.add_argument('--max-accuracy', type=float, default=0.65, help='最小準確度閾值')
    parser.add_argument('--export', choices=['csv', 'excel'], default=None, help='導出格式')
    parser.add_argument('--sort-by', choices=['auc', 'accuracy', 'f1'], default='auc', help='排序依據')
    parser.add_argument('--results-file', type=str, default='./data/models/training_results.json', help='結果文件路徑')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    if results is None:
        return
    
    # Print summary
    print_summary(results)
    
    # Print detailed results
    print_detailed_results(results, sort_by=args.sort_by)
    
    # Check models that need retraining
    print_needs_retraining(results, min_auc=args.min_auc, max_accuracy=args.max_accuracy)
    
    # Export if requested
    if args.export:
        export_results(results, format=args.export)


if __name__ == '__main__':
    main()
