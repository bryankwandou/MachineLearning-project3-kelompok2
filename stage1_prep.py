"""Stage 1: Preprocessing + Scaling + PCA + EDA plots"""
import os, gc, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import joblib
warnings.filterwarnings('ignore')

BASE   = r"e:\TUGAS KULIAH ANTIGRAVITY\MachineLearning-project3-kelompok2"
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
REP    = os.path.join(BASE, "reports")
for d in [DATA, MODELS, REP]: os.makedirs(d, exist_ok=True)

CSV = r"E:\DATASET TUGAS KULIAH\Credit Card Dataset for Clustering\CC GENERAL.csv"
print("Reading dataset...")
df_raw = pd.read_csv(CSV)
print(f"  Shape: {df_raw.shape}")

# Drop ID, impute
df = df_raw.drop(columns=['CUST_ID'])
FEATURES = list(df.columns)
imputer = SimpleImputer(strategy='median')
X_imp = imputer.fit_transform(df)
df_imp = pd.DataFrame(X_imp, columns=FEATURES)
joblib.dump(imputer, os.path.join(MODELS, 'imputer.pkl'))
df_imp.to_csv(os.path.join(DATA, 'preprocessed.csv'), index=False)
print(f"  preprocessed.csv saved, {df_imp.shape}")

# Scale
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_imp)
joblib.dump(scaler, os.path.join(MODELS, 'scaler.pkl'))
print("  scaler.pkl saved")

# PCA 2D for visualization
pca2 = PCA(n_components=2, random_state=42)
X_pca = pca2.fit_transform(X_sc)
joblib.dump(pca2, os.path.join(MODELS, 'pca.pkl'))
pd.DataFrame(X_pca, columns=['PC1','PC2']).to_csv(
    os.path.join(DATA, 'pca_result.csv'), index=False)
print("  pca.pkl + pca_result.csv saved")

# Variance (first 8 components)
pca_full = PCA(n_components=8, random_state=42)
pca_full.fit(X_sc)
var_ratio = pca_full.explained_variance_ratio_
pca_var_2d = float(np.sum(var_ratio[:2]) * 100)
print(f"  PCA PC1+PC2 variance: {pca_var_2d:.2f}%")

# Save intermediate state
import json
state = {'pca_var_2d': round(pca_var_2d,4), 'n_rows': len(df_imp),
         'n_features': len(FEATURES), 'features': FEATURES}
with open(os.path.join(BASE, '_state.json'), 'w') as f: json.dump(state, f)

# ── Plot 01: Distribution 9 features ──────────────────────────
print("Plotting 01_distribusi_fitur...")
feats9 = ['BALANCE','PURCHASES','CASH_ADVANCE','CREDIT_LIMIT','PAYMENTS',
          'MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE','PURCHASES_FREQUENCY']
fig, axes = plt.subplots(3, 3, figsize=(11, 7))
for ax, col in zip(axes.flatten(), feats9):
    ax.hist(df_imp[col], bins=35, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(col, fontsize=8, fontweight='bold')
    ax.set_xlabel('Nilai', fontsize=7); ax.set_ylabel('Freq', fontsize=7)
    ax.tick_params(labelsize=7)
fig.suptitle('Distribusi Fitur Dataset Credit Card Customer', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REP,'01_distribusi_fitur.png'), dpi=120, bbox_inches='tight')
plt.close('all'); gc.collect()
print("  01 done")

# ── Plot 02: Missing values ────────────────────────────────────
print("Plotting 02_missing_values...")
mv = df_raw.isnull().sum(); mv = mv[mv>0].sort_values(ascending=True)
if len(mv):
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(mv)*1.1)))
    colors = ['red' if v>100 else 'orange' for v in mv.values]
    bars = ax.barh(mv.index, mv.values, color=colors, edgecolor='white')
    for b,v in zip(bars, mv.values):
        ax.text(b.get_width()+1, b.get_y()+b.get_height()/2, str(v), va='center', fontsize=10)
    ax.set_title('Analisis Missing Values sebelum Preprocessing', fontsize=11, fontweight='bold')
    ax.set_xlabel('Jumlah Missing Values')
    plt.tight_layout()
    plt.savefig(os.path.join(REP,'02_missing_values.png'), dpi=120, bbox_inches='tight')
    plt.close('all'); gc.collect()
print("  02 done")

# ── Plot 11: PCA Variance ──────────────────────────────────────
print("Plotting 11_pca_variance...")
pcs = [f'PC{i+1}' for i in range(8)]
vals = [v*100 for v in var_ratio]
fig, ax = plt.subplots(figsize=(8, 4))
cmap = ['#1a237e' if i<2 else '#5c6bc0' for i in range(8)]
ax.bar(pcs, vals, color=cmap, edgecolor='white', zorder=3)
ax.plot(pcs, np.cumsum(vals), 'r--o', linewidth=2, label='Kumulatif', zorder=4)
for i,(p,v) in enumerate(zip(pcs,vals)):
    ax.text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold')
ax.set_title('PCA — Explained Variance per Principal Component', fontsize=11, fontweight='bold')
ax.set_ylabel('Variance Explained (%)'); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REP,'11_pca_variance.png'), dpi=120, bbox_inches='tight')
plt.close('all'); gc.collect()
print("  11 done")

print("\nSTAGE 1 COMPLETE")
