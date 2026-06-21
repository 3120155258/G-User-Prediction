"""
运行脚本：检测依赖并执行主程序
"""
import sys
import os
import subprocess

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs')
if os.path.exists(LIB_PATH):
    sys.path.insert(0, LIB_PATH)

def check_dependencies():
    """检查必要的库是否已安装"""
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
    }
    
    missing = []
    for import_name, pkg_name in required.items():
        try:
            __import__(import_name)
            print(f"  [OK] {pkg_name}")
        except ImportError:
            missing.append(pkg_name)
            print(f"  [MISSING] {pkg_name}")
    
    if missing:
        print(f"\n缺少以下依赖: {', '.join(missing)}")
        print("正在尝试自动安装...")
        for pkg in missing:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', pkg])
        print("依赖安装完成，请重新运行此脚本。")
        sys.exit(1)
    
    print("\n所有依赖已就绪!")

if __name__ == '__main__':
    print("=" * 50)
    print("5G 用户预测 - 依赖检查")
    print("=" * 50)
    check_dependencies()
    
    print("\n" + "=" * 50)
    print("开始执行主程序...")
    print("=" * 50)
    
    # 执行主脚本
    main_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
    subprocess.run([sys.executable, main_script], check=True)
