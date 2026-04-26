import pandas as pd
import numpy as np

df = pd.read_csv('germany_ndvi_era5_lc_monthly_raw.csv')
df['year'] = df['year'].astype(int)
df['month'] = df['month'].astype(int)

# 1. Climatology: her adm1 x lc_name x month için 2017-2024 mean ve std
clim = df.groupby(['adm1', 'lc_name', 'month'])[['ndvi','temp_c','soil_moisture','precip_mm']].agg(['mean','std']).reset_index()
clim.columns = ['adm1','lc_name','month',
                'ndvi_mean','ndvi_std',
                'temp_mean','temp_std',
                'sm_mean','sm_std',
                'precip_mean','precip_std']

df = df.merge(clim, on=['adm1','lc_name','month'], how='left')

# 2. Anomaliler (z-score)
df['ndvi_anomaly']   = (df['ndvi']         - df['ndvi_mean'])  / df['ndvi_std']
df['temp_anomaly']   = (df['temp_c']       - df['temp_mean'])  / df['temp_std']
df['sm_anomaly']     = (df['soil_moisture'] - df['sm_mean'])   / df['sm_std']
df['precip_anomaly'] = (df['precip_mm']    - df['precip_mean'])/ df['precip_std']

# 3. Stress label (iki eşik)
df['stress_1sd']   = (df['ndvi_anomaly'] < -1.0).astype(int)
df['stress_20pct'] = df.groupby(['adm1','lc_name','month'])['ndvi_anomaly'].transform(
    lambda x: (x < x.quantile(0.20)).astype(int)
)

# 4. Sort + lag üretimi (adm1 x lc_name bazında zaman sırası)
df = df.sort_values(['adm1','lc_name','year','month']).reset_index(drop=True)

lag_vars = ['sm_anomaly','temp_anomaly','precip_anomaly','ndvi_anomaly']
for var in lag_vars:
    for lag in [1, 2, 3]:
        df[f'{var}_lag{lag}'] = df.groupby(['adm1','lc_name'])[var].shift(lag)

# 5. lag olan satırları tut (ilk 3 ay her grup için NaN)
df_model = df.dropna().copy()

print('Final shape:', df_model.shape)
print('Stress_1sd rate:', df_model['stress_1sd'].mean().round(3))
print('Stress_20pct rate:', df_model['stress_20pct'].mean().round(3))

df_model.to_csv('germany_model_ready.csv', index=False)
print('Saved: germany_model_ready.csv')
