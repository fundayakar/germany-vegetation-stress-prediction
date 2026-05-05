import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# Temporal split: 2017-2021 train, 2022-2024 test
train = df[df['year'] <= 2021]
test  = df[df['year'] >= 2022]

X_tr = train[feature_cols].values
y_tr = train['stress_1sd'].values
X_te = test[feature_cols].values
y_te = test['stress_1sd'].values

scale_pw = (y_tr==0).sum() / (y_tr==1).sum()
model = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw,
                           random_state=42, eval_metric='logloss', verbosity=0)
model.fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:,1]
pred  = (proba > 0.5).astype(int)

auc = roc_auc_score(y_te, proba)
ap  = average_precision_score(y_te, proba)
f1  = f1_score(y_te, pred, zero_division=0)

print("Temporal transferability: train 2017-2021, test 2022-2024")
print(f"Test stress rate: {y_te.mean():.3f}")
print(f"ROC-AUC: {auc:.3f}")
print(f"PR-AUC:  {ap:.3f}")
print(f"F1:      {f1:.3f}")

print("\nPer year performance (2022-2024):")
print(f"{'Year':<6} {'N':>5} {'stress%':>8} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8}")
print('-'*48)
for yr in [2022, 2023, 2024]:
    idx = test['year'] == yr
    if idx.sum() < 10:
        continue
    y_yr = y_te[idx.values]
    p_yr = proba[idx.values]
    if y_yr.sum() < 2:
        print(f"{yr:<6} {idx.sum():>5} {y_yr.mean():>8.3f} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        continue
    print(f"{yr:<6} {idx.sum():>5} {y_yr.mean():>8.3f} "
          f"{roc_auc_score(y_yr, p_yr):>8.3f} "
          f"{average_precision_score(y_yr, p_yr):>8.3f} "
          f"{f1_score(y_yr, (p_yr>0.5).astype(int), zero_division=0):>8.3f}")

# Figure
colors = ['#3a7ebf', '#e6a817', '#2d7a4f']
width = 0.25
categories = ['Spatial CV\n(2017-2024)', 'Temporal transfer\n(2022-2024)', '2022', '2023', '2024']
aucs_t = [0.905, auc, 0.688, 0.554, 0.569]
aps_t  = [0.701, ap,  0.501, 0.223, 0.066]
f1s_t  = [0.646, f1,  0.297, 0.189, 0.050]

x = np.arange(len(categories))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - width, aucs_t, width, label='ROC-AUC', color=colors[0], edgecolor='none')
ax.bar(x,         aps_t,  width, label='PR-AUC',  color=colors[1], edgecolor='none')
ax.bar(x + width, f1s_t,  width, label='F1',      color=colors[2], edgecolor='none')
for i, (a, p, f) in enumerate(zip(aucs_t, aps_t, f1s_t)):
    ax.text(i - width, a + 0.005, f'{a:.3f}', ha='center', va='bottom', fontsize=7)
    ax.text(i,         p + 0.005, f'{p:.3f}', ha='center', va='bottom', fontsize=7)
    ax.text(i + width, f + 0.005, f'{f:.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel('Score', fontsize=10)
ax.set_title('Temporal transferability: train 2017-2021, test 2022-2024', fontsize=11, pad=10)
ax.spines[['top','right']].set_visible(False)
ax.legend(fontsize=9, frameon=False)
ax.axvline(1.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(0.05, 0.95, 'Reference', fontsize=7.5, color='gray')
ax.text(1.65, 0.95, 'Temporal transfer', fontsize=7.5, color='gray')
plt.tight_layout()
plt.savefig('fig_temporal_transfer.png', dpi=180, bbox_inches='tight')
plt.close()
print("Saved: fig_temporal_transfer.png")
