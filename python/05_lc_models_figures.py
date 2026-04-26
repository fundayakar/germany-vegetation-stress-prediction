import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import shap
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
le_adm = LabelEncoder()
df['adm_enc'] = le_adm.fit_transform(df['adm1'])
feature_cols_lc = feature_cols + ['adm_enc']

label_map = {
    'ndvi_anomaly_lag1':   'NDVI anomaly (t-1)',
    'precip_anomaly_lag1': 'Precip anomaly (t-1)',
    'temp_anomaly_lag2':   'Temp anomaly (t-2)',
    'sm_anomaly_lag1':     'SM anomaly (t-1)',
    'temp_anomaly_lag3':   'Temp anomaly (t-3)',
    'sm_anomaly_lag2':     'SM anomaly (t-2)',
    'precip_anomaly_lag3': 'Precip anomaly (t-3)',
    'month':               'Month',
    'precip_anomaly_lag2': 'Precip anomaly (t-2)',
    'temp_anomaly_lag1':   'Temp anomaly (t-1)',
    'sm_anomaly_lag3':     'SM anomaly (t-3)',
    'adm_enc':             'Region',
}

lc_types  = ['cropland', 'forest', 'grassland']
lc_colors = {'cropland': '#e6a817', 'forest': '#2d7a4f', 'grassland': '#7ec8a0'}

states = df['adm1'].unique()
np.random.seed(42)
np.random.shuffle(states)
state_folds = np.array_split(states, 4)

def spatial_cv_lc(df_lc, feature_cols, state_folds):
    X = df_lc[feature_cols].values
    y = df_lc['stress_1sd'].values
    scale_pw = max((y==0).sum() / max((y==1).sum(), 1), 1)
    aucs, aps, f1s = [], [], []
    for fold_states in state_folds:
        test_idx  = df_lc['adm1'].isin(fold_states).values
        train_idx = ~test_idx
        if y[train_idx].sum() < 5 or y[test_idx].sum() < 2:
            continue
        model = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw,
                                   random_state=42, eval_metric='logloss', verbosity=0)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:,1]
        pred  = (proba > 0.5).astype(int)
        aucs.append(roc_auc_score(y[test_idx], proba))
        aps.append(average_precision_score(y[test_idx], proba))
        f1s.append(f1_score(y[test_idx], pred, zero_division=0))
    return np.mean(aucs), np.mean(aps), np.mean(f1s)

print(f"{'Land cover':<12} {'N':>6} {'stress%':>8} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8}")
print('-'*56)
perf = {}
lc_models = {}
for lc in lc_types:
    df_lc = df[df['lc_name'] == lc].copy()
    auc, ap, f1 = spatial_cv_lc(df_lc, feature_cols_lc, state_folds)
    perf[lc] = {'ROC-AUC': auc, 'PR-AUC': ap, 'F1': f1}
    print(f"{lc:<12} {len(df_lc):>6} {df_lc['stress_1sd'].mean():>8.3f} {auc:>8.3f} {ap:>8.3f} {f1:>8.3f}")
    X_full = df_lc[feature_cols_lc].values
    y_full = df_lc['stress_1sd'].values
    scale_pw = (y_full==0).sum() / (y_full==1).sum()
    m = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw,
                           random_state=42, eval_metric='logloss', verbosity=0)
    m.fit(X_full, y_full)
    lc_models[lc] = m

# Figure 1: performance comparison
metrics = ['ROC-AUC', 'PR-AUC', 'F1']
x = np.arange(len(metrics))
width = 0.25
fig, ax = plt.subplots(figsize=(6.5, 4))
for i, lc in enumerate(lc_types):
    vals = [perf[lc][m] for m in metrics]
    ax.bar(x + i*width, vals, width, label=lc.capitalize(), color=lc_colors[lc], edgecolor='none')
    for j, v in enumerate(vals):
        ax.text(x[j] + i*width, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score', fontsize=10)
ax.set_title('Model performance by land cover (spatial CV)', fontsize=11, pad=10)
ax.spines[['top','right']].set_visible(False)
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig('fig_lc_performance.png', dpi=180, bbox_inches='tight')
plt.close()

# Figure 2: top 3 SHAP per LC
shap_top3 = {}
for lc in lc_types:
    df_lc = df[df['lc_name'] == lc].copy()
    X_full = df_lc[feature_cols_lc]
    explainer = shap.TreeExplainer(lc_models[lc])
    sv = explainer.shap_values(X_full)
    top3 = pd.Series(np.abs(sv).mean(axis=0), index=feature_cols_lc).sort_values(ascending=False).head(3)
    shap_top3[lc] = list(top3.items())

fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
for ax, lc in zip(axes, lc_types):
    feats  = [label_map.get(f, f) for f, _ in shap_top3[lc]]
    values = [v for _, v in shap_top3[lc]]
    ax.barh(feats[::-1], values[::-1], color=lc_colors[lc], edgecolor='none', height=0.55)
    ax.set_title(lc.capitalize(), fontsize=10, pad=6)
    ax.set_xlabel('Mean |SHAP|', fontsize=8.5)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
plt.suptitle('Top 3 predictors by land cover type', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('fig_lc_shap_top3.png', dpi=180, bbox_inches='tight')
plt.close()

# Figure 3: stress probability map (cropland, mean across 2017-2024)
le_lc_full = LabelEncoder()
df['lc_enc'] = le_lc_full.fit_transform(df['lc_name'])
fc_full = feature_cols + ['lc_enc', 'adm_enc']
X_full = df[fc_full].values
y_full = df['stress_1sd'].values
scale_pw = (y_full==0).sum() / (y_full==1).sum()
model_full = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw,
                                random_state=42, eval_metric='logloss', verbosity=0)
model_full.fit(X_full, y_full)
df['stress_prob'] = model_full.predict_proba(X_full)[:,1]

map_df = df[df['lc_name']=='cropland'].groupby('adm1')['stress_prob'].mean().reset_index()
map_df = map_df.sort_values('stress_prob', ascending=True)

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.barh(map_df['adm1'], map_df['stress_prob'],
        color=plt.cm.YlOrRd(map_df['stress_prob'] / map_df['stress_prob'].max()),
        edgecolor='none', height=0.7)
ax.set_xlabel('Mean predicted stress probability', fontsize=10)
ax.set_title('Predicted vegetation stress probability\nby federal state (cropland, 2017-2024)', fontsize=10, pad=8)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(axis='y', labelsize=8.5)
ax.tick_params(axis='x', labelsize=8.5)
ax.axvline(0.15, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Mean stress rate')
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig('fig_stress_prob_map.png', dpi=180, bbox_inches='tight')
plt.close()

print("Saved: fig_lc_performance.png, fig_lc_shap_top3.png, fig_stress_prob_map.png")
