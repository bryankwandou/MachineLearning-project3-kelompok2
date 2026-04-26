"""Stage 2: K-Means — find K optimal + train final model"""
import os, gc, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
from kneed import KneeLocator
warnings.filterwarnings('ignore')

BASE   = r"e:\TUGAS KULIAH ANTIGRAVITY\MachineLearning-project3-kelompok2"
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
REP    = os.path.join(BASE, "reports")

print("Loading scaled data...")
df_imp = pd.read_csv(os.path.join(DATA, 'preprocessed.csv'))
scaler = joblib.load(os.path.join(MODELS, 'scaler.pkl'))
X_sc   = scaler.transform(df_imp.values)
print(f"  X_sc shape: {X_sc.shape}")

# ── Elbow + Silhouette sweep K=2..10 ──────────────────────────
print("Running K sweep (K=2..10)...")
K_RANGE = range(2, 11)
inertias, sil_scores = [], []
SAMPLE = min(3000, len(X_sc))

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    lbl = km.fit_predict(X_sc)
    inertias.append(float(km.inertia_))
    s = silhouette_score(X_sc, lbl, sample_size=SAMPLE, random_state=42)
    sil_scores.append(float(s))
    print(f"  K={k}: inertia={km.inertia_:.0f}  sil={s:.4f}")
    del km, lbl; gc.collect()

# Determine optimal K
kl = KneeLocator(list(K_RANGE), inertias, curve='convex', direction='decreasing')
K_ELBOW = kl.elbow if kl.elbow else 4
K_SIL   = list(K_RANGE)[int(np.argmax(sil_scores))]
K_OPT   = K_SIL
print(f"\nK_elbow={K_ELBOW}  K_silhouette={K_SIL}  -> K_OPTIMAL={K_OPT}")

# ── Plot 03: Elbow ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(K_RANGE), inertias, 'b-o', linewidth=2, markersize=6)
ax.axvline(x=K_OPT, color='red', linestyle='--', linewidth=2)
mid_y = (min(inertias) + max(inertias)) / 2
ax.text(K_OPT+0.15, mid_y, f'K optimal = {K_OPT}', color='red', fontsize=10)
ax.set_title('Elbow Method — Penentuan K Optimal', fontsize=11, fontweight='bold')
ax.set_xlabel('Jumlah Cluster (K)'); ax.set_ylabel('Inertia'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REP,'03_elbow_method.png'), dpi=120, bbox_inches='tight')
plt.close('all'); gc.collect()
print("03_elbow_method.png saved")

# ── Plot 04: Silhouette Score ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(K_RANGE), sil_scores, 'g-o', linewidth=2, markersize=6)
ax.axvline(x=K_SIL, color='red', linewidth=2)
ax.set_title('Silhouette Score per K — K-Means', fontsize=11, fontweight='bold')
ax.set_xlabel('Jumlah Cluster (K)'); ax.set_ylabel('Silhouette Score'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REP,'04_silhouette_score.png'), dpi=120, bbox_inches='tight')
plt.close('all'); gc.collect()
print("04_silhouette_score.png saved")

# ── Final K-Means training ─────────────────────────────────────
print(f"Training K-Means final K={K_OPT}...")
km_final = KMeans(n_clusters=K_OPT, random_state=42, n_init=10, max_iter=300)
km_labels = km_final.fit_predict(X_sc)
KM_INERTIA = float(km_final.inertia_)
KM_SIL     = float(silhouette_score(X_sc, km_labels, sample_size=min(5000,len(X_sc)), random_state=42))
KM_DB      = float(davies_bouldin_score(X_sc, km_labels))
km_dist    = {str(k): int(v) for k,v in pd.Series(km_labels).value_counts().sort_index().items()}
print(f"  Inertia={KM_INERTIA:.2f}  Silhouette={KM_SIL:.4f}  DB={KM_DB:.4f}")
print(f"  Distribution: {km_dist}")
joblib.dump(km_final, os.path.join(MODELS, 'kmeans_final.pkl'))

# ── Plot 06: K-Means Scatter ───────────────────────────────────
pca2 = joblib.load(os.path.join(MODELS, 'pca.pkl'))
X_pca = np.load if False else pd.read_csv(os.path.join(DATA, 'pca_result.csv')).values
centroids_pca = pca2.transform(km_final.cluster_centers_)
PAL = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#999999']

fig, ax = plt.subplots(figsize=(9, 6))
for cid in range(K_OPT):
    mask = km_labels == cid
    ax.scatter(X_pca[mask,0], X_pca[mask,1], c=PAL[cid], alpha=0.45, s=10, label=f'Cluster {cid}')
ax.scatter(centroids_pca[:,0], centroids_pca[:,1], marker='*', s=280, c='black', zorder=5, label='Centroid')
ax.set_title(f'Hasil K-Means Clustering (K={K_OPT})\nSilhouette = {KM_SIL:.4f}', fontsize=11, fontweight='bold')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REP,'06_kmeans_scatter.png'), dpi=120, bbox_inches='tight')
plt.close('all'); gc.collect()
print("06_kmeans_scatter.png saved")

# Save state
with open(os.path.join(BASE, '_state.json'), 'r') as f: state = json.load(f)
state.update({'K_OPT': K_OPT, 'KM_INERTIA': KM_INERTIA, 'KM_SIL': KM_SIL,
              'KM_DB': KM_DB, 'km_dist': km_dist, 'km_labels': km_labels.tolist()})
with open(os.path.join(BASE, '_state.json'), 'w') as f: json.dump(state, f)

print("\nSTAGE 2 COMPLETE")
