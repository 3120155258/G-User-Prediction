"""
=============================================================================
5G 用户预测 - 完整解决方案
=============================================================================
基于用户基本信息和通信数据，预测用户是否为5G用户。
实现多种ML模型：LightGBM, XGBoost, Random Forest, Logistic Regression
评估指标：AUC
=============================================================================
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs'))
libs_path = os.path.abspath("./libs")
os.environ["PYTHONPATH"] = libs_path
sys.path.insert(0, libs_path)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
import xgboost as xgb

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# 配置
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'train.csv')
FIGURES_PATH = os.path.join(BASE_DIR, 'figures')
MODELS_PATH = os.path.join(BASE_DIR, 'models')
os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

# ============================================================================
# 第一部分：数据加载与探索性分析 (EDA)
# ============================================================================
print("=" * 70)
print("第一部分：数据加载与探索性分析 (EDA)")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"数据集形状: {df.shape}")
print(f"样本数: {df.shape[0]}, 特征数: {df.shape[1] - 2}")  # 减去id和target

# 1.1 目标变量分布
target_counts = df['target'].value_counts()
print(f"\n目标变量分布:")
print(f"  非5G用户 (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.2f}%)")
print(f"  5G用户 (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(['Non-5G (0)', '5G (1)'], [target_counts.get(0,0), target_counts.get(1,0)],
            color=['#3498db', '#e74c3c'])
axes[0].set_title('Target Distribution')
for i, v in enumerate([target_counts.get(0,0), target_counts.get(1,0)]):
    axes[0].text(i, v + max(target_counts)*0.01, str(v), ha='center', fontsize=11, fontweight='bold')
axes[1].pie([target_counts.get(0,0), target_counts.get(1,0)], labels=['Non-5G', '5G'],
            autopct='%1.2f%%', colors=['#3498db', '#e74c3c'], explode=(0, 0.05))
axes[1].set_title('Target Proportion')
plt.savefig(os.path.join(FIGURES_PATH, '01_target_distribution.png'))
plt.close()
print("  [图] target_distribution.png 已保存")

# 1.2 特征分类
cat_cols = [c for c in df.columns if c.startswith('cat_')]
num_cols = [c for c in df.columns if c.startswith('num_')]
print(f"\n特征分类:")
print(f"  离散型特征 (cat_0~cat_19): {len(cat_cols)} 个")
print(f"  数值型特征 (num_0~num_37): {len(num_cols)} 个")

# 1.3 缺失值检查
missing = df.isnull().sum()
print(f"\n缺失值: 总计 {missing.sum()} 个")
if missing.sum() > 0:
    print(missing[missing > 0])

# 1.4 特征与目标相关性分析
print("\n计算特征与target的相关系数...")
correlations = {}
for c in df.columns:
    if c not in ['id', 'target']:
        try:
            corr = abs(df[c].corr(df['target']))
            if not np.isnan(corr):
                correlations[c] = corr
        except:
            pass

top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:20]
print("与target相关性最高的20个特征:")
for name, corr in top_features:
    print(f"  {name}: {corr:.4f}")

# 相关性可视化
top_corr_features = [c for c, _ in top_features[:15]]
corr_matrix = df[top_corr_features + ['target']].corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, ax=ax, linewidths=0.5)
ax.set_title('Top 15 Feature Correlations', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(FIGURES_PATH, '02_correlation_heatmap.png'))
plt.close()
print("  [图] correlation_heatmap.png 已保存")

# ============================================================================
# 第二部分：数据预处理
# ============================================================================
print("\n" + "=" * 70)
print("第二部分：数据预处理")
print("=" * 70)

# 分离特征和标签
X = df.drop(columns=['id', 'target'])
y = df['target']
feature_names = X.columns.tolist()

# 2.1 处理缺失值
if X.isnull().sum().sum() > 0:
    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode()[0] if len(X[c].mode()) > 0 else -1)
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())

# 2.2 离散特征编码 (Label Encoding)
print("对离散特征进行Label Encoding...")
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c].astype(str))
    label_encoders[c] = le
print(f"  已编码 {len(cat_cols)} 个离散特征")

# 2.3 异常值处理 (Winsorize - clip极端值到1%和99%分位数)
print("处理数值特征极端值...")
for c in num_cols:
    lower = X[c].quantile(0.01)
    upper = X[c].quantile(0.99)
    X[c] = X[c].clip(lower, upper)

# 2.4 特征筛选：移除低方差和低相关特征
print("进行特征筛选...")
# 移除去重后类别数过少的离散特征 (< 2)
low_variance_cats = [c for c in cat_cols if X[c].nunique() < 2]
# 移除常量数值特征（唯一值数量 <= 1）
constant_nums = [c for c in num_cols if X[c].nunique() <= 1]
if constant_nums:
    print(f"  发现 {len(constant_nums)} 个常量数值特征: {constant_nums}")
# 移除与目标相关性极低的特征 (< 0.0005) 或相关性为 NaN 的常量特征
low_corr_features = [k for k, v in correlations.items() if v < 0.0005 or np.isnan(v)]
features_to_drop = list(set(low_variance_cats + constant_nums + low_corr_features))
features_to_drop = [f for f in features_to_drop if f in X.columns]

if len(features_to_drop) > 0:
    print(f"  删除 {len(features_to_drop)} 个低质量特征: {features_to_drop}")
    X = X.drop(columns=features_to_drop)
    # 更新列名
    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in num_cols if c in X.columns]
else:
    print("  所有特征均保留")

print(f"最终特征数: {X.shape[1]} (离散:{len(cat_cols)}, 数值:{len(num_cols)})")

# 2.5 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")
print(f"训练集正样本比例: {y_train.mean():.4f}")
print(f"验证集正样本比例: {y_val.mean():.4f}")

# ============================================================================
# 第三部分：模型训练与评估
# ============================================================================
print("\n" + "=" * 70)
print("第三部分：模型训练与评估")
print("=" * 70)

results = {}
models = {}
feature_importances = {}

# ---- 模型1: Logistic Regression (基线模型) ----
print("\n[模型1] Logistic Regression (基线模型)")
from sklearn.pipeline import Pipeline
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
])
lr.fit(X_train, y_train)
y_pred_proba_lr = lr.predict_proba(X_val)[:, 1]
auc_lr = roc_auc_score(y_val, y_pred_proba_lr)
results['Logistic Regression'] = auc_lr
models['Logistic Regression'] = lr
print(f"  AUC: {auc_lr:.6f}")

# ---- 模型2: Random Forest ----
print("\n[模型2] Random Forest")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_leaf=50,
    random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
)
rf.fit(X_train, y_train)
y_pred_proba_rf = rf.predict_proba(X_val)[:, 1]
auc_rf = roc_auc_score(y_val, y_pred_proba_rf)
results['Random Forest'] = auc_rf
models['Random Forest'] = rf
feature_importances['Random Forest'] = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(f"  AUC: {auc_rf:.6f}")

# ---- 模型3: XGBoost ----
print("\n[模型3] XGBoost")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    random_state=RANDOM_STATE, n_jobs=-1, eval_metric='auc',
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
auc_xgb = roc_auc_score(y_val, y_pred_proba_xgb)
results['XGBoost'] = auc_xgb
models['XGBoost'] = xgb_model
feature_importances['XGBoost'] = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(f"  AUC: {auc_xgb:.6f}")

# ---- 模型4: LightGBM ----
print("\n[模型4] LightGBM")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=10, learning_rate=0.03,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    class_weight='balanced'
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
)
y_pred_proba_lgb = lgb_model.predict_proba(X_val)[:, 1]
auc_lgb = roc_auc_score(y_val, y_pred_proba_lgb)
results['LightGBM'] = auc_lgb
models['LightGBM'] = lgb_model
feature_importances['LightGBM'] = pd.Series(lgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(f"  AUC: {auc_lgb:.6f}")

# ---- 模型5: 集成模型 (Voting) ----
print("\n[模型5] 集成模型 (Voting Classifier)")
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                   random_state=RANDOM_STATE, n_jobs=-1, eval_metric='auc')),
        ('lgb', lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.03,
                                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, 
                                       random_state=RANDOM_STATE, n_jobs=-1))
    ],
    voting='soft',
    weights=[2, 2, 1]
)
voting_clf.fit(X_train, y_train)
y_pred_proba_vote = voting_clf.predict_proba(X_val)[:, 1]
auc_vote = roc_auc_score(y_val, y_pred_proba_vote)
results['Ensemble (Voting)'] = auc_vote
models['Ensemble (Voting)'] = voting_clf
print(f"  AUC: {auc_vote:.6f}")

# ============================================================================
# 第四部分：模型比较与可视化
# ============================================================================
print("\n" + "=" * 70)
print("第四部分：模型比较与可视化")
print("=" * 70)

# 4.1 AUC结果汇总
print("\nAUC 结果汇总:")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for rank, (name, auc) in enumerate(sorted_results, 1):
    marker = " [最佳]" if rank == 1 else ""
    print(f"  {rank}. {name}: {auc:.6f}{marker}")

# 4.2 ROC曲线对比
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
for i, (name, model) in enumerate(models.items()):
    if name != 'Ensemble (Voting)':
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        y_prob = y_pred_proba_vote
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc_score = results[name]
    ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
            label=f'{name} (AUC={auc_score:.5f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.savefig(os.path.join(FIGURES_PATH, '03_roc_curves.png'))
plt.close()
print("  [图] roc_curves.png 已保存")

# 4.3 AUC柱状图对比
fig, ax = plt.subplots(figsize=(10, 6))
names_list = [n for n, _ in sorted_results]
aucs_list = [a for _, a in sorted_results]
bar_colors = ['#e74c3c' if i == len(aucs_list)-1 else '#3498db' for i in range(len(aucs_list))]
bars = ax.barh(names_list, aucs_list, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, aucs_list):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
            f'{val:.5f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('AUC Score', fontsize=12)
ax.set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
ax.set_xlim([min(aucs_list)*0.99, max(aucs_list)*1.005])
ax.grid(axis='x', alpha=0.3)
plt.savefig(os.path.join(FIGURES_PATH, '04_auc_comparison.png'))
plt.close()
print("  [图] auc_comparison.png 已保存")

# 4.4 特征重要性分析 (最佳模型)
best_model_name = sorted_results[0][0]
print(f"\n最佳模型: {best_model_name}")

if best_model_name in feature_importances:
    fi = feature_importances[best_model_name]
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 30
    top_fi = fi.head(top_n)
    colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))
    ax.barh(range(top_n), top_fi.values, color=colors_fi[::-1], edgecolor='white', linewidth=1)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_fi.index)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances ({best_model_name})', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.savefig(os.path.join(FIGURES_PATH, '05_feature_importance.png'))
    plt.close()
    print(f"  [图] feature_importance.png 已保存")

# 4.5 特征重要性比较 (多模型)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_names_for_fi = ['XGBoost', 'LightGBM', 'Random Forest']
for idx, mn in enumerate(model_names_for_fi):
    if mn in feature_importances:
        fi = feature_importances[mn].head(15)
        axes[idx].barh(range(15), fi.values, color=colors[idx], edgecolor='white')
        axes[idx].set_yticks(range(15))
        axes[idx].set_yticklabels(fi.index, fontsize=7)
        axes[idx].invert_yaxis()
        axes[idx].set_title(f'{mn}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Importance')
plt.suptitle('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, '06_feature_importance_comparison.png'))
plt.close()
print("  [图] feature_importance_comparison.png 已保存")

# ============================================================================
# 第五部分：交叉验证验证
# ============================================================================
print("\n" + "=" * 70)
print("第五部分：交叉验证 (K-Fold)")
print("=" * 70)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# 对最佳模型进行交叉验证
cv_models = {
    'LightGBM': lgb.LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.03,
                                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                  random_state=RANDOM_STATE, n_jobs=-1, eval_metric='auc'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                             random_state=RANDOM_STATE, n_jobs=-1),
}

for name, model in cv_models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, 
                                 scoring='roc_auc', n_jobs=-1, verbose=0)
    print(f"  {name}:")
    print(f"    CV AUC (mean ± std): {cv_scores.mean():.5f} ± {cv_scores.std():.5f}")
    print(f"    各折AUC: {[f'{s:.5f}' for s in cv_scores]}")

# ============================================================================
# 第六部分：功能拓展 - 超参数优化（部分）
# ============================================================================
print("\n" + "=" * 70)
print("第六部分：功能拓展 - 超参数优化 (GridSearchCV)")
print("=" * 70)

# 在部分参数空间上进行搜索
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.03, 0.05],
}
print("对XGBoost进行网格搜索...")
base_xgb = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='auc')
grid_search = GridSearchCV(
    base_xgb, param_grid, cv=3, scoring='roc_auc',
    n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
print(f"  最佳参数: {grid_search.best_params_}")
print(f"  最佳CV AUC: {grid_search.best_score_:.5f}")

# 使用最佳参数评估
best_xgb = grid_search.best_estimator_
y_pred_proba_best_xgb = best_xgb.predict_proba(X_val)[:, 1]
auc_best_xgb = roc_auc_score(y_val, y_pred_proba_best_xgb)
print(f"  验证集 AUC: {auc_best_xgb:.5f}")

# 更新结果
results['XGBoost (Tuned)'] = auc_best_xgb
models['XGBoost (Tuned)'] = best_xgb

# ============================================================================
# 第七部分：最终总结
# ============================================================================
print("\n" + "=" * 70)
print("第七部分：最终结果总结")
print("=" * 70)

final_sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)
print(f"\n{'模型名称':<25} {'AUC Score':<12}")
print("-" * 40)
for name, auc in final_sorted:
    print(f"{name:<25} {auc:.6f}")

best_name, best_auc = final_sorted[0]
print(f"\n最佳模型: {best_name} (AUC = {best_auc:.6f})")
print(f"\n所有图表已保存至: {FIGURES_PATH}")
print(f"模型已训练完成!")

print("\n" + "=" * 70)
print("解决方案完成！")
print("=" * 70)
