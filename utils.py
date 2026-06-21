"""
工具模块：路径配置、数据工具函数
"""
import sys
import os

# 设置库路径
LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs')
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

# 全局配置
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.csv')
FIGURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
