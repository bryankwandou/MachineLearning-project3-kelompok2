"""Stage 3: DBSCAN — find eps optimal + train final model"""
import os, gc, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
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
X_pca  = pd.read_csv(os.path.join(DATA, 'pca_result.csv')).values
print(f"  X_sc shape: {X_sc.shape}")

MIN_SAMPLES = 5

# ── K-distance graph ──────────────────────────────────────────
print("Computing k-distance (NearestNeighbors)...")
nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X_sc)
distances, _ = nbrs.kneighbors(X_sc)
k_dist = np.sort(distances[:, MIN_SAMPLES-1])[::-1]
del nbrs, distances; gc.collect()

kl_eps = KneeLocator(range(len(k_dist)), k_dist, curve='convex', direction='decreasing')
knee_idx = kl_eps.knee if kl_eps.knee else int(len(k_dist)*0.02)
EPS_BASE = float(k_dist[knee_idx])
print(f"  eps from knee: {EPS_BASE:.4f}")

# ── Plot 05: K-Distance Graph ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(k_dist, 'b-', linewidth=1)
ax.scatter(knee_idx, EPS_BASE, color='red', s=120, zorder=5, label=f'eps={EPS_BASE:.3f}')
ax.axhline(y=EPS_BASE, color='red', linestyle='--', linewidth=1.5)
ax.set_title('K-Distance Graph — Penentuan eps Optimal DBSCAN', fontsize=11, fontweight='bold')
ax.set_xlabel('Data Points (diurutkan)'); ax.set_ylabel(f'Jarak ke Tetangga ke-{MIN_SAMPLES}')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REP,'05_kdistance_dbscan.png'), dpi=120, bbox_inches='tight')
plt.close('all'); gc.collect()
print("05_kdistance_dbscan.png saved")

# ── Sweep 5 eps variants ──────────────────────────────────────
print(f"\n{'eps':>10} {'N_cl':>7} {'Noise%':>8} {'Silhouette':>12}")
print("-"*44)
SAMPLE = min(3000, len(X_sc))
best = None
for mult in [0.70, 0.85, 1.00, 1.20, 1.50]:
    eps_try = EPS_BASE * mult
    lbl = DBSCAN(eps=eps_try, min_samples=MIN_SAMPLES).fit_predict(X_sc)
    n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
    n_noise = int(np.sum(lbl == -1))
    pct_noise = n_noise / len(lbl) * 100
    nm = lbl != -1
    sil = silhouette_score(X_sc[nm], lbl[nm], sample_size=SAMPLE, random_state=42) \
          if n_cl >= 2 and nm.sum() > 10 else -99.0
    print(f"{eps_try:>10.4f} {n_cl:>7} {pct_noise:>7.1f}% {sil:>12.4f}")
    if n_cl >= 2 and pct_noise < 30 and sil > -99:
        if best is None or sil > best['sil']:
            best = {'eps': eps_try, 'sil': sil, 'n_cl': n_cl, 'pct_noise': pct_noise}
    del lbl; gc.collect()

if best is None:
    best = {'eps': EPS_BASE * 1.2, 'sil': 0.0, 'n_cl': 1, 'pct_noise': 50.0}
EPS_FINAL = best['eps']
print(f"\n-> eps_final={EPS_FINAL:.4f}")

# ── Train final DBSCAN ────────────────────────────────────────
print(f"Training DBSCAN eps={EPS_FINAL:.4f}...")
db_final  = DBSCAN(eps=EPS_FINAL, min_samples=MIN_SAMPLES)
db_labels = db_final.fit_predict(X_sc)
DB_CLUSTERS = len(set(db_labels)) - (1 if -1 in db_labels else 0)
DB_NOISE    = int(np.sum(db_labels == -1))
DB_NOISE_PCT= float(DB_NOISE / len(db_labels) * 100)
nm = db_labels != -1
DB_SIL = float(silhouette_score(X_sc[nm], db_labels[nm],
               sample_size=min(5000,nm.sum()), random_state=42)) \
         if DB_CLUSTERS >= 2 and nm.sum() > 10 else 0.0
print(f"  Clusters={DB_CLUSTERS}  Noise={DB_NOISE}({DB_NOISE_PCT:.1f}%)  Sil={DB_SIL:.4f}")
joblib.dump(db_final, os.path.join(MODELS, 'dbscan_final.pkl'))

# ── Plot 07: DBSCAN Scatter ────────────────────────────────────
PAL = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#999999','#00CED1','#FFD700']
db_unique = sorted(set(db_labels))
color_map = {lbl: ('black' if lbl==-1 else PAL[i % len(PAL)]) for i, lbl in enumerate(l for l in db_unique if l!=-1)}
color_map[-1] = 'black'

fig, ax = plt.subplots(figsize=(9, 6))
for lbl in db_unique:
    mask = db_labels == lbl
    if lbl == -1:
        ax.scatter(X_pca[mask,0], X_pca[mask,1], c='black', alpha=0.12, s=7, marker='x', label=f'Noise ({DB_NOISE})')
    else:
        ax.scatter(X_pca[mask,0], X_pca[mask,1], c=color_map[lbl], alpha=0.45, s=10, label=f'Cluster {lbl}')
ax.set_title(f'Hasil DBSCAN Clustering (eps={EPS_FINAL:.3f})\n{DB_CLUSTERS} Clusters | {DB_NOISE} Noise', fontsize=11, fontweight='bold')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REP,'07_dbscan_scatter.png'), dpi=120, bbox_inches='tight')
plt.close('all'); gc.collect()
print("07_dbscan_scatter.png saved")

# Update state
with open(os.path.join(BASE, '_state.json'), 'r') as f: state = json.load(f)
state.update({'EPS_FINAL': round(EPS_FINAL,4), 'MIN_SAMPLES': MIN_SAMPLES,
              'DB_CLUSTERS': DB_CLUSTERS, 'DB_NOISE': DB_NOISE,
              'DB_NOISE_PCT': round(DB_NOISE_PCT,2), 'DB_SIL': round(DB_SIL,4),
              'db_labels': db_labels.tolist()})
with open(os.path.join(BASE, '_state.json'), 'w') as f: json.dump(state, f)

print("\nSTAGE 3 COMPLETE")
