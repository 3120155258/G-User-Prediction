# 5G User Prediction - AI Group Assignment

## Project Overview

This project implements a machine learning solution to predict whether a user is a 5G user based on user profile and communication data. The task is a binary classification problem using the AUC metric for evaluation.

**Course**: Artificial Intelligence  
**Department**: Software Engineering  
**Due**: Week 16

## Dataset Description

The dataset `train.csv` contains 60 columns:
- `id`: Sample identifier
- `cat_0` ~ `cat_19`: Categorical features (20 columns)
- `num_0` ~ `num_37`: Numerical features (38 columns)
- `target`: Prediction target - whether user is a 5G user (0 or 1)

## Implemented Models

1. **Logistic Regression** - Baseline model
2. **Random Forest** - Ensemble tree model with class balancing
3. **XGBoost** - Gradient boosting with regularization
4. **LightGBM** - Efficient gradient boosting with early stopping
5. **Ensemble (Voting Classifier)** - Soft voting ensemble of XGBoost + LightGBM + Random Forest

## Extended Features

- Feature correlation analysis and selection
- Extreme value processing (Winsorization)
- K-Fold cross validation
- Grid search hyperparameter optimization
- Multi-dimensional visualization:
  - Target distribution analysis
  - Feature correlation heatmap
  - ROC curve comparison
  - AUC bar chart comparison
  - Feature importance analysis across models

## Project Structure

```
.
├── train.csv                    # Training dataset
├── 5G_User_Prediction.ipynb     # Jupyter Notebook (main deliverable)
├── main.py                      # Complete Python solution script
├── eda_analysis.py              # Exploratory data analysis script
├── utils.py                     # Utility module
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── figures/                     # Generated visualizations
├── models/                      # Saved model files
└── libs/                        # Python package directory
```

## Requirements

- Python 3.8+
- pandas, numpy, scipy
- scikit-learn
- matplotlib, seaborn
- xgboost, lightgbm

## Installation

### Option 1: Install to system Python
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn xgboost lightgbm
```

### Option 2: Install to local libs directory (offline)
```bash
pip install --target=./libs pandas numpy scipy scikit-learn matplotlib seaborn xgboost lightgbm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## How to Run

### Method 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook 5G_User_Prediction.ipynb
```

### Method 2: Python Script
```bash
# Set Python path to include local packages
set PYTHONPATH=./libs
python main.py
```

### Method 3: Using the provided run script
```bash
python run.py
```

## Evaluation

The primary evaluation metric is **AUC (Area Under ROC Curve)**. Sample evaluation code:

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred_proba)
print(f'AUC: {auc}')
```

## Results

The best performing model achieves high AUC on the validation set. Detailed results and comparisons are generated during execution and saved to the `figures/` directory.

## Team Members

(To be filled by the group)

## License

This project is for educational purposes as part of the AI course assignment.

## Acknowledgments

- Dataset provided by the Software Engineering Department
- Libraries: scikit-learn, XGBoost, LightGBM
