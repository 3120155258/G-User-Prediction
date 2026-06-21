"""
数据集探索性分析脚本
分析 train.csv 的基本特征：规模、分布、缺失值、相关性等
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# 加载数据
print("=" * 60)
print("1. 数据加载")
print("=" * 60)
df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
print(f"数据集形状: {df.shape}")
print(f"列数: {df.shape[1]}")
print(f"样本数: {df.shape[0]}")
print(f"\n列名: {list(df.columns)}")

# 基本统计
print("\n" + "=" * 60)
print("2. 数据类型与基本信息")
print("=" * 60)
print(df.dtypes.value_counts())
print(f"\n内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 目标变量分布
print("\n" + "=" * 60)
print("3. 目标变量(target)分布")
print("=" * 60)
target_counts = df['target'].value_counts()
print(f"target=0 (非5G用户): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.2f}%)")
print(f"target=1 (5G用户): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.2f}%)")

# 可视化目标分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(['非5G用户 (0)', '5G用户 (1)'], 
            [target_counts.get(0, 0), target_counts.get(1, 0)],
            color=['steelblue', 'coral'])
axes[0].set_title('Target Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate([target_counts.get(0, 0), target_counts.get(1, 0)]):
    axes[0].text(i, v + 100, str(v), ha='center', fontsize=11)

axes[1].pie([target_counts.get(0, 0), target_counts.get(1, 0)], 
            labels=['非5G用户', '5G用户'],
            autopct='%1.2f%%', colors=['steelblue', 'coral'],
            explode=(0, 0.05))
axes[1].set_title('Target Proportion')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("已保存: figures/target_distribution.png")

# 缺失值分析
print("\n" + "=" * 60)
print("4. 缺失值分析")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
if len(missing_df) > 0:
    print(missing_df)
else:
    print("数据集中无缺失值!")

# 特征的缺失情况
print(f"\n存在缺失值的列数: {(missing > 0).sum()}")

# 特征分类分析
print("\n" + "=" * 60)
print("5. 特征类型分析")
print("=" * 60)
cat_cols = [c for c in df.columns if c.startswith('cat_')]
num_cols = [c for c in df.columns if c.startswith('num_')]
print(f"离散型特征 (cat_): {len(cat_cols)} 个")
print(f"  列名: {cat_cols}")
print(f"数值型特征 (num_): {len(num_cols)} 个")
print(f"  列名: {num_cols}")

# 离散特征分析
print("\n" + "=" * 60)
print("6. 离散型特征分析")
print("=" * 60)
for c in cat_cols[:5]:
    n_unique = df[c].nunique()
    print(f"{c}: {n_unique} 个唯一值, 最常见值: {df[c].mode().values[:3]}")

# 数值特征统计
print("\n" + "=" * 60)
print("7. 数值型特征统计")
print("=" * 60)
num_stats = df[num_cols].describe()
print(num_stats)

# 检查0值列
zero_cols = (df[num_cols] == 0).sum()
zero_cols_high = zero_cols[zero_cols > len(df) * 0.5]
if len(zero_cols_high) > 0:
    print(f"\n零值超过50%的数值特征 ({len(zero_cols_high)} 个):")
    for c, v in zero_cols_high.items():
        print(f"  {c}: {v/len(df)*100:.1f}% 为零")

# 检查常数列
constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
if constant_cols:
    print(f"\n常数列: {constant_cols}")

# 相关性分析
print("\n" + "=" * 60)
print("8. 特征与目标相关性分析")
print("=" * 60)

# 计算特征与target的相关系数
correlations = {}
for c in df.columns:
    if c not in ['id', 'target']:
        try:
            corr = df[c].corr(df['target'])
            if not np.isnan(corr):
                correlations[c] = abs(corr)
        except:
            pass

# Top特征
top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:20]
print("与target相关性最高的20个特征:")
for name, corr in top_features:
    print(f"  {name}: {corr:.4f}")

# 零相关性特征
zero_corr = [k for k, v in correlations.items() if v < 0.0001]
print(f"\n与target几乎零相关的特征数: {len(zero_corr)}")

# 目标变量与关键特征的分布
print("\n" + "=" * 60)
print("9. 生成特征分布可视化")
print("=" * 60)

# 数值特征分布（选取相关性高的特征）
top_num = [c for c, _ in top_features if c.startswith('num_')][:6]
if top_num:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(top_num):
        for target_val in [0, 1]:
            subset = df[df['target'] == target_val][col]
            axes[i].hist(subset, bins=50, alpha=0.6, 
                        label=f'target={target_val}', density=True)
        axes[i].set_title(f'{col} Distribution by Target')
        axes[i].legend()
        axes[i].set_xlabel(col)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'num_features_by_target.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: figures/num_features_by_target.png")

# 离散特征分布
top_cat = [c for c, _ in top_features if c.startswith('cat_')][:6]
if top_cat:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(top_cat):
        cross = pd.crosstab(df[col], df['target'], normalize='index')
        cross.plot(kind='bar', stacked=True, ax=axes[i], 
                   color=['steelblue', 'coral'])
        axes[i].set_title(f'{col} vs Target')
        axes[i].set_xlabel(col)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cat_features_by_target.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存: figures/cat_features_by_target.png")

# 相关系数热力图
top_corr_features = [c for c, _ in top_features[:15]]
corr_matrix = df[top_corr_features + ['target']].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, ax=ax,
            xticklabels=True, yticklabels=True)
ax.set_title('Top 15 Features Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("已保存: figures/correlation_heatmap.png")

print("\n" + "=" * 60)
print("分析完成!")
print("=" * 60)
