import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
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

X = df[feature_cols]
y = df['stress_1sd']

scale_pw = (y==0).sum() / (y==1).sum()
model = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pw,
                           random_state=42, eval_metric='logloss', verbosity=0)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Global mean |SHAP| per feature
shap_df = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("Global SHAP feature importance:")
print(shap_df.to_string(index=False))

# Land cover bazli SHAP
print("\nSHAP by land cover (top 5 features per LC):")
for lc in ['cropland','forest','grassland']:
    idx = df['lc_name'] == lc
    sv = np.abs(shap_values[idx]).mean(axis=0)
    top = pd.Series(sv, index=feature_cols).sort_values(ascending=False).head(5)
    print(f"\n  {lc}:")
    for feat, val in top.items():
        print(f"    {feat:<30} {val:.4f}")

# SHAP değerlerini kaydet
shap_out = pd.DataFrame(shap_values, columns=feature_cols)
shap_out['lc_name'] = df['lc_name'].values
shap_out['adm1']    = df['adm1'].values
shap_out['year']    = df['year'].values
shap_out['month']   = df['month'].values
shap_out.to_csv('shap_values.csv', index=False)
shap_df.to_csv('shap_importance.csv', index=False)
print("\nSaved: shap_values.csv, shap_importance.csv")
