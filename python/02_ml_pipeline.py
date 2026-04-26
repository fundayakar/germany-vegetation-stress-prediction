import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('germany_model_ready.csv')

feature_cols = [
    'sm_anomaly_lag1','sm_anomaly_lag2','sm_anomaly_lag3',
    'temp_anomaly_lag1','temp_anomaly_lag2','temp_anomaly_lag3',
    'precip_anomaly_lag1','precip_anomaly_lag2','precip_anomaly_lag3',
    'ndvi_anomaly_lag1', 'month'
]
le_lc  = LabelEncoder()
le_adm = LabelEncoder()
df['lc_enc']  = le_lc.fit_transform(df['lc_name'])
df['adm_enc'] = le_adm.fit_transform(df['adm1'])
feature_cols += ['lc_enc', 'adm_enc']

target = 'stress_1sd'
X = df[feature_cols].values
y = df[target].values

# Spatial blocked CV: 16 eyaleti 4 fold'a böl
states = df['adm1'].unique()
np.random.seed(42)
np.random.shuffle(states)
state_folds = np.array_split(states, 4)

def spatial_cv(model, X, y, df, state_folds):
    aucs, aps, f1s = [], [], []
    for fold_states in state_folds:
        test_idx  = df['adm1'].isin(fold_states).values
        train_idx = ~test_idx
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:,1]
        pred  = (proba > 0.5).astype(int)
        aucs.append(roc_auc_score(y[test_idx], proba))
        aps.append(average_precision_score(y[test_idx], proba))
        f1s.append(f1_score(y[test_idx], pred, zero_division=0))
    return np.mean(aucs), np.mean(aps), np.mean(f1s)

scale_pw = (y==0).sum() / (y==1).sum()

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost':             xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw, random_state=42, eval_metric='logloss', verbosity=0)
}

print(f"{'Model':<22} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8}")
print('-'*52)
for name, model in models.items():
    auc, ap, f1 = spatial_cv(model, X, y, df, state_folds)
    print(f"{name:<22} {auc:>8.3f} {ap:>8.3f} {f1:>8.3f}")

# XGBoost random CV karşılaştırma
xgb2 = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw, random_state=42, eval_metric='logloss', verbosity=0)
cv = cross_validate(xgb2, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring=['roc_auc','average_precision'], n_jobs=-1)
print(f"\nXGBoost random CV:   ROC-AUC={cv['test_roc_auc'].mean():.3f}  PR-AUC={cv['test_average_precision'].mean():.3f}")
print("Spatial CV daha düşükse: leakage yok, güvenilir sonuç.")
