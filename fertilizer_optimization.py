import os
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import matplotlib.pyplot as plt

# Paths
OUTDIR = './outputs_fert_opt'
os.makedirs(OUTDIR, exist_ok=True)
DATA_PATH = 'fert_opt_synthetic.csv'

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

# --- Feature engineering ---
if {'soil_N', 'soil_P', 'soil_K'}.issubset(df.columns):
    df['N_deficit'] = 120 - df['soil_N']  # assume 120 kg/ha recommended N
    df['P_deficit'] = 60 - df['soil_P']
    df['K_deficit'] = 40 - df['soil_K']
df['fert_cost_ratio'] = df['fert_cost_per_ha'] / df['price_per_ton']

# One-hot encode upfront
df_model = pd.get_dummies(df, columns=['fert_type', 'crop', 'soil_texture'], drop_first=True)
target_col = 'yield_kg_ha'
exclude_cols = ['plot_id', 'yield_kg_ha', 'price_per_ton', 'fert_cost_per_ha']
feature_cols = [c for c in df_model.columns if c not in exclude_cols]

# Prepare X, y
X = df_model[feature_cols]
y = df_model[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Train model ---
lgb_train = lgb.Dataset(X_train_s, label=y_train)
lgb_eval = lgb.Dataset(X_test_s, label=y_test, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'learning_rate': 0.03,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'seed': 42
}

print("Training LightGBM...")
bst = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_train, lgb_eval],
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=20)]
)

y_pred = bst.predict(X_test_s, num_iteration=bst.best_iteration)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE={rmse:.1f}, R2={r2:.3f}")

# --- Optimization with batch prediction ---
fert_products = [
    {'name': 'Urea', 'N_pct': 46.0, 'P_pct': 0.0, 'K_pct': 0.0, 'cost_per_kg': 0.3},
    {'name': 'DAP', 'N_pct': 18.0, 'P_pct': 46.0, 'K_pct': 0.0, 'cost_per_kg': 0.45},
    {'name': 'NPK_20_20_0', 'N_pct': 20.0, 'P_pct': 20.0, 'K_pct': 0.0, 'cost_per_kg': 0.4}
]
dose_grid = np.arange(0, 301, 20)

candidates = []
for idx, row in df.iterrows():
    for prod in fert_products:
        for d in dose_grid:
            applied_N = prod['N_pct'] / 100.0 * d
            applied_P = prod['P_pct'] / 100.0 * d
            applied_K = prod['K_pct'] / 100.0 * d
            rec = row.copy()
            rec['fert_dose'] = d
            rec['applied_N'] = applied_N
            rec['applied_P'] = applied_P
            rec['applied_K'] = applied_K
            rec['prod_name'] = prod['name']
            rec['prod_cost'] = prod['cost_per_kg']
            candidates.append(rec)

cand_df = pd.DataFrame(candidates)
cand_enc = pd.get_dummies(cand_df, columns=['fert_type', 'crop', 'soil_texture'], drop_first=True)
for c in feature_cols:
    if c not in cand_enc.columns:
        cand_enc[c] = 0
cand_enc = cand_enc[feature_cols]
cand_s = scaler.transform(cand_enc)

pred_yields = bst.predict(cand_s, num_iteration=bst.best_iteration)
cand_df['pred_yield'] = pred_yields
cand_df['profit'] = (cand_df['price_per_ton'] * (cand_df['pred_yield'] / 1000.0)) - (cand_df['fert_dose'] * cand_df['prod_cost'])

# choose best per plot
rec_df = cand_df.sort_values(['plot_id', 'profit'], ascending=[True, False]).groupby('plot_id').head(1)
rec_df[['plot_id', 'prod_name', 'fert_dose', 'pred_yield', 'profit']].to_csv(os.path.join(OUTDIR, 'recommendations.csv'), index=False)
print('Saved recommendations:', os.path.join(OUTDIR, 'recommendations.csv'))

# Plot profit distribution
plt.figure(figsize=(6, 3))
plt.hist(rec_df['profit'], bins=20)
plt.xlabel('Profit (INR/ha)')
plt.ylabel('Count')
plt.title('Distribution of recommended profit levels')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'profit_hist.png'))
print('Saved plot:', os.path.join(OUTDIR, 'profit_hist.png'))

print("Done.")
