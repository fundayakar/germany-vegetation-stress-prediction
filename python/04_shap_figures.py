import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# --- data + model ---
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

X = df[feature_cols]
y = df['stress_1sd']
scale_pw = (y==0).sum() / (y==1).sum()
model = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw,
                           random_state=42, eval_metric='logloss', verbosity=0)
model.fit(X, y)
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

label_map = {
    'ndvi_anomaly_lag1':   'NDVI anomaly (t-1)',
    'precip_anomaly_lag1': 'Precip anomaly (t-1)',
    'temp_anomaly_lag2':   'Temp anomaly (t-2)',
    'sm_anomaly_lag1':     'Soil moisture anomaly (t-1)',
    'temp_anomaly_lag3':   'Temp anomaly (t-3)',
    'sm_anomaly_lag2':     'Soil moisture anomaly (t-2)',
    'precip_anomaly_lag3': 'Precip anomaly (t-3)',
    'month':               'Month',
    'precip_anomaly_lag2': 'Precip anomaly (t-2)',
    'temp_anomaly_lag1':   'Temp anomaly (t-1)',
    'sm_anomaly_lag3':     'Soil moisture anomaly (t-3)',
    'lc_enc':              'Land cover class',
    'adm_enc':             'Region',
}

shap_df = pd.DataFrame({
    'feature': feature_cols,
    'label':   [label_map.get(f, f) for f in feature_cols],
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=True)

def feat_color(f):
    if 'ndvi' in f:   return '#2d7a4f'
    if 'sm' in f:     return '#3a7ebf'
    if 'precip' in f: return '#5b4ea8'
    if 'temp' in f:   return '#c0392b'
    return '#888888'

colors = [feat_color(f) for f in shap_df['feature']]

# Figure 1: global feature importance bar
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(shap_df['label'], shap_df['mean_abs_shap'], color=colors, height=0.65, edgecolor='none')
ax.set_xlabel('Mean |SHAP value|', fontsize=10)
ax.set_title('Feature importance (XGBoost, spatial CV)', fontsize=11, pad=10)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(axis='y', labelsize=8.5)
ax.tick_params(axis='x', labelsize=8.5)
legend_items = [
    mpatches.Patch(color='#2d7a4f', label='NDVI'),
    mpatches.Patch(color='#3a7ebf', label='Soil moisture'),
    mpatches.Patch(color='#5b4ea8', label='Precipitation'),
    mpatches.Patch(color='#c0392b', label='Temperature'),
    mpatches.Patch(color='#888888', label='Other'),
]
ax.legend(handles=legend_items, fontsize=8, loc='lower right', frameon=False)
plt.tight_layout()
plt.savefig('fig_shap_importance.png', dpi=180, bbox_inches='tight')
plt.close()

# Figure 2: land cover SHAP comparison
lc_list = ['cropland', 'forest', 'grassland']
top_feats_global = shap_df.sort_values('mean_abs_shap', ascending=False)['feature'].iloc[:8].tolist()

lc_shap = {}
for lc in lc_list:
    idx = df['lc_name'] == lc
    lc_shap[lc] = np.abs(shap_values[idx]).mean(axis=0)

lc_df = pd.DataFrame(lc_shap, index=feature_cols).loc[top_feats_global]
lc_df.index = [label_map.get(f, f) for f in lc_df.index]
lc_df = lc_df.iloc[::-1]

x = np.arange(len(lc_df))
width = 0.25
lc_colors = {'cropland': '#e6a817', 'forest': '#2d7a4f', 'grassland': '#7ec8a0'}

fig, ax = plt.subplots(figsize=(7, 5.5))
for i, lc in enumerate(lc_list):
    ax.barh(x + i*width, lc_df[lc], width, label=lc.capitalize(),
            color=lc_colors[lc], edgecolor='none')
ax.set_yticks(x + width)
ax.set_yticklabels(lc_df.index, fontsize=8.5)
ax.set_xlabel('Mean |SHAP value|', fontsize=10)
ax.set_title('Feature importance by land cover type', fontsize=11, pad=10)
ax.spines[['top','right']].set_visible(False)
ax.legend(fontsize=8.5, frameon=False)
ax.tick_params(axis='x', labelsize=8.5)
plt.tight_layout()
plt.savefig('fig_shap_landcover.png', dpi=180, bbox_inches='tight')
plt.close()

# Figure 3: beeswarm (top 10)
top10_idx = [feature_cols.index(f) for f in
             shap_df.sort_values('mean_abs_shap', ascending=False)['feature'].iloc[:10]]
shap_top   = shap_values[:, top10_idx]
X_top      = X.iloc[:, top10_idx]
feat_labels = [label_map.get(feature_cols[i], feature_cols[i]) for i in top10_idx]

fig, ax = plt.subplots(figsize=(7, 5.5))
shap.summary_plot(shap_top, X_top, feature_names=feat_labels,
                  show=False, plot_size=None, color_bar_label='Feature value')
plt.title('SHAP beeswarm (top 10 features)', fontsize=11, pad=10)
plt.tight_layout()
plt.savefig('fig_shap_beeswarm.png', dpi=180, bbox_inches='tight')
plt.close()

print("Saved: fig_shap_importance.png, fig_shap_landcover.png, fig_shap_beeswarm.png")
