#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理转换后的模型目录 - 删除旧的不正常的转换结果

使用方式:
  python scripts/cleanup_converted_models.py
"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup():
    """清理转换的模型目录"""
    converted_dir = Path('./data/models_converted')
    
    if not converted_dir.exists():
        logger.info(f"{converted_dir} does not exist. Nothing to clean.")
        return
    
    logger.info(f"Cleaning up {converted_dir}...")
    
    try:
        # 删除整个目录
        shutil.rmtree(str(converted_dir))
        logger.info(f"✓ Deleted: {converted_dir}")
        
        # 重新创建空目录
        converted_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created empty directory: {converted_dir}")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("Cleanup complete! Now you can re-run the conversion.")
    logger.info("="*60)
    logger.info("\nNext step: python scripts/convert_model_to_new_format.py --all\n")
    
    return True

if __name__ == '__main__':
    cleanup()
