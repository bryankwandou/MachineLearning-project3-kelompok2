"""Stage 4: Comparison plots + heatmap + CSV outputs + metadata + text reports"""
import os, gc, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

BASE   = r"e:\TUGAS KULIAH ANTIGRAVITY\MachineLearning-project3-kelompok2"
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
REP    = os.path.join(BASE, "reports")

# Load state
with open(os.path.join(BASE, '_state.json'), 'r') as f: S = json.load(f)
K_OPT       = S['K_OPT'];       KM_SIL  = S['KM_SIL']
KM_DB       = S['KM_DB'];       KM_INERTIA = S['KM_INERTIA']
EPS_FINAL   = S['EPS_FINAL'];   DB_CLUSTERS = S['DB_CLUSTERS']
DB_NOISE    = S['DB_NOISE'];    DB_NOISE_PCT= S['DB_NOISE_PCT']
DB_SIL      = S['DB_SIL'];     MIN_SAMPLES = S['MIN_SAMPLES']
km_labels   = np.array(S['km_labels'])
db_labels   = np.array(S['db_labels'])
km_dist     = S['km_dist']
pca_var_2d  = S['pca_var_2d']
N_ROWS      = S['n_rows']

df_imp  = pd.read_csv(os.path.join(DATA, 'preprocessed.csv'))
X_pca   = pd.read_csv(os.path.join(DATA, 'pca_result.csv')).values
scaler  = joblib.load(os.path.join(MODELS, 'scaler.pkl'))
km_final= joblib.load(os.path.join(MODELS, 'kmeans_final.pkl'))
pca2    = joblib.load(os.path.join(MODELS, 'pca.pkl'))
X_sc    = scaler.transform(df_imp.values)

centroids_pca = pca2.transform(km_final.cluster_centers_)
PAL     = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#999999']
DB_PAL  = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#999999','#00CED1','#FFD700']
db_unique  = sorted(set(db_labels))
db_cmap = {}
ci = 0
for lbl in db_unique:
    db_cmap[lbl] = 'black' if lbl == -1 else DB_PAL[ci % len(DB_PAL)]; ci += (0 if lbl==-1 else 1)

# ── Plot 08: Side by side ──────────────────────────────────────
print("Plotting 08_perbandingan_side_by_side...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
ax = axes[0]
for cid in range(K_OPT):
    m = km_labels==cid
    ax.scatter(X_pca[m,0], X_pca[m,1], c=PAL[cid], alpha=0.4, s=8, label=f'Cluster {cid}')
ax.scatter(centroids_pca[:,0], centroids_pca[:,1], marker='*', s=250, c='black', zorder=5)
ax.set_title(f'K-Means (K={K_OPT})\nSilhouette={KM_SIL:.4f}', fontsize=12, fontweight='bold')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
for lbl in db_unique:
    m = db_labels==lbl
    if lbl==-1:
        ax.scatter(X_pca[m,0], X_pca[m,1], c='black', alpha=0.12, s=7, marker='x', label='Noise')
    else:
        ax.scatter(X_pca[m,0], X_pca[m,1], c=db_cmap[lbl], alpha=0.4, s=8, label=f'Cluster {lbl}')
ax.set_title(f'DBSCAN (eps={EPS_FINAL:.3f})\n{DB_CLUSTERS} Clusters | Sil={DB_SIL:.4f}', fontsize=12, fontweight='bold')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
fig.suptitle('Perbandingan K-Means vs DBSCAN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REP,'08_perbandingan_side_by_side.png'), dpi=120, bbox_inches='tight')
plt.close('all'); gc.collect()
print("  08 done")

# ── Plot 09: Heatmap cluster profile ──────────────────────────
print("Plotting 09_cluster_heatmap...")
HF = ['BALANCE','PURCHASES','CASH_ADVANCE','CREDIT_LIMIT','PAYMENTS','PRC_FULL_PAYMENT','TENURE']
df_h = df_imp.copy(); df_h['cluster'] = km_labels
profile = df_h.groupby('cluster')[HF].mean()
norm = profile.copy()
for c in norm.columns:
    mn,mx = norm[c].min(), norm[c].max()
    norm[c] = (norm[c]-mn)/(mx-mn) if mx>mn else 0.5
fig, ax = plt.subplots(figsize=(11, max(4, K_OPT*1.1)))
sns.heatmap(norm, annot=profile.round(1), fmt='g', cmap='YlOrRd',
            linewidths=0.5, ax=ax, annot_kws={'size':8})
ax.set_title('Profil Karakteristik Rata-rata per Cluster (K-Means)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REP,'09_cluster_heatmap.png'), dpi=120, bbox_inches='tight')
plt.close('all'); gc.collect()
print("  09 done")

# ── Plot 10: Pie charts ────────────────────────────────────────
print("Plotting 10_distribusi_cluster...")
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
km_cnt = pd.Series(km_labels).value_counts().sort_index()
axes[0].pie(km_cnt.values, labels=[f'Cluster {i}\n({v})' for i,v in km_cnt.items()],
            colors=PAL[:len(km_cnt)], autopct='%1.1f%%', startangle=140)
axes[0].set_title(f'K-Means (K={K_OPT})', fontsize=11, fontweight='bold')
db_cnt = pd.Series(db_labels).value_counts().sort_index()
db_pie_v, db_pie_l, db_pie_c = [], [], []
for lbl,cnt in db_cnt.items():
    db_pie_v.append(cnt)
    db_pie_l.append(f'Noise({cnt})' if lbl==-1 else f'Cluster {lbl}\n({cnt})')
    db_pie_c.append(db_cmap[lbl])
axes[1].pie(db_pie_v, labels=db_pie_l, colors=db_pie_c, autopct='%1.1f%%', startangle=140)
axes[1].set_title(f'DBSCAN (eps={EPS_FINAL:.3f})', fontsize=11, fontweight='bold')
fig.suptitle('Distribusi Nasabah per Cluster', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REP,'10_distribusi_cluster.png'), dpi=120, bbox_inches='tight')
plt.close('all'); gc.collect()
print("  10 done")

# ── CSV: hasil_clustering ──────────────────────────────────────
print("Saving hasil_clustering.csv...")
df_hasil = df_imp.copy()
df_hasil['KMeans_Label'] = km_labels
df_hasil['DBSCAN_Label'] = db_labels
df_hasil['PC1'] = X_pca[:,0]
df_hasil['PC2'] = X_pca[:,1]
df_hasil.to_csv(os.path.join(DATA,'hasil_clustering.csv'), index=False)
print(f"  {df_hasil.shape}")

# ── metadata.json ─────────────────────────────────────────────
pemenang = 'K-Means' if KM_SIL >= DB_SIL else 'DBSCAN'
meta = {
    "nama_proyek": "Project 3 ML Kelompok B",
    "dataset": "Credit Card Customer Segmentation",
    "jumlah_baris": N_ROWS,
    "jumlah_fitur": 17,
    "pca_variance_explained_pct": pca_var_2d,
    "kmeans": {
        "k_optimal": K_OPT, "silhouette_score": round(KM_SIL,4),
        "davies_bouldin_score": round(KM_DB,4), "inertia": round(KM_INERTIA,2),
        "distribusi_cluster": km_dist
    },
    "dbscan": {
        "eps_optimal": EPS_FINAL, "min_samples": MIN_SAMPLES,
        "jumlah_cluster": DB_CLUSTERS, "jumlah_noise": DB_NOISE,
        "persen_noise": DB_NOISE_PCT, "silhouette_score": round(DB_SIL,4)
    },
    "pemenang": pemenang,
    "alasan_pemenang": (f"{pemenang} menang Silhouette lebih tinggi "
                        f"(K-Means:{KM_SIL:.4f} vs DBSCAN:{DB_SIL:.4f}). "
                        f"DBSCAN unggul deteksi {DB_NOISE} outlier.")
}
with open(os.path.join(MODELS,'metadata.json'),'w',encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print("metadata.json saved")

# ── interpretasi_bisnis.txt ────────────────────────────────────
print("Writing interpretasi_bisnis.txt...")
def label_cl(row):
    if row['PRC_FULL_PAYMENT']>0.35 and row['PURCHASES']>3000: return "Nasabah Premium — Pembayar Lunas Aktif"
    elif row['CASH_ADVANCE']>2000 and row['PURCHASES']<500:    return "Nasabah Berisiko — Pengguna Cash Advance"
    elif row['BALANCE']>3000 and row['PRC_FULL_PAYMENT']<0.1: return "Nasabah Revolving — Terjebak Cicilan"
    else: return "Nasabah Pasif — Pengguna Kartu Jarang"

lines = ["="*65,"INTERPRETASI BISNIS HASIL CLUSTERING K-MEANS",
         "Dataset: Credit Card Customer Segmentation",
         f"Jumlah Nasabah Total: {N_ROWS}","="*65+"\n"]

for cid in range(K_OPT):
    mask = km_labels==cid
    row  = profile.loc[cid]
    n    = mask.sum()
    pct  = n/len(km_labels)*100
    nama = label_cl(row)
    lines += [f"\n{'═'*65}",f"CLUSTER {cid} — {nama}",
              f"Jumlah Nasabah: {n} ({pct:.1f}% dari total)","═"*65,
              "\nKARAKTERISTIK NUMERIK:"]
    descs = {
        'BALANCE':          ("Saldo hutang TINGGI" if row['BALANCE']>3000 else "Saldo RENDAH"),
        'PURCHASES':        ("Belanja AKTIF" if row['PURCHASES']>1000 else "Belanja JARANG"),
        'CASH_ADVANCE':     ("Tarik tunai SERING" if row['CASH_ADVANCE']>1000 else "Tarik tunai JARANG"),
        'CREDIT_LIMIT':     ("Limit TINGGI" if row['CREDIT_LIMIT']>7000 else "Limit RENDAH"),
        'PRC_FULL_PAYMENT': ("Bayar LUNAS SERING" if row['PRC_FULL_PAYMENT']>0.3 else "Bayar lunas JARANG"),
        'PAYMENTS':         ("Total bayar TINGGI" if row['PAYMENTS']>3000 else "Total bayar RENDAH"),
        'TENURE':           (f"Nasabah selama {row['TENURE']:.0f} bulan"),
    }
    for feat, desc in descs.items():
        val = row[feat] if feat in row.index else df_imp[feat][mask].mean()
        lines.append(f"  {feat:<30}: {val:>10.2f}  → {desc}")

    lines.append("\nPROFIL BISNIS:")
    if "Premium" in nama:
        lines += ["  Segmen paling menguntungkan bagi bank — disiplin bayar lunas,",
                  "  belanja aktif, limit tinggi karena rekam jejak baik.",
                  "  Tidak bergantung cash advance; menggunakan kartu untuk transaksi nyata."]
    elif "Berisiko" in nama:
        lines += ["  Tanda tekanan keuangan: menarik tunai lebih sering daripada belanja.",
                  "  Risiko gagal bayar dan NPL lebih tinggi dari segmen lain.",
                  "  Bunga cash advance memperparah posisi keuangan mereka."]
    elif "Revolving" in nama:
        lines += ["  Saldo hutang besar tapi hanya bayar minimum — terjebak bunga berputar.",
                  "  Bank mendapat revenue bunga jangka pendek namun risiko NPL meningkat.",
                  "  Perlu intervensi untuk mencegah gagal bayar total."]
    else:
        lines += ["  Hampir tidak pernah menggunakan kartu — nilai transaksi minimal.",
                  "  Kontribusi revenue rendah; peluang untuk diaktifkan melalui promo.",
                  "  Jika tidak diaktifkan dalam 12 bulan pertimbangkan penutupan akun."]

    lines.append("\nREKOMENDASI BISNIS:")
    if "Premium" in nama:
        lines += ["  1. Cashback & reward points berlipat ganda.",
                  "  2. Upgrade limit kartu secara proaktif.",
                  "  3. Akses lounge bandara dan privilege perjalanan."]
    elif "Berisiko" in nama:
        lines += ["  1. Batasi fasilitas cash advance secara bertahap.",
                  "  2. Tawarkan pinjaman personal bunga rendah sebagai alternatif.",
                  "  3. Edukasi biaya tersembunyi cash advance via notifikasi."]
    elif "Revolving" in nama:
        lines += ["  1. Program cicilan 0% untuk melunasi saldo berputar.",
                  "  2. Konsultasi keuangan gratis via aplikasi mobile banking.",
                  "  3. Restrukturisasi hutang dengan insentif bunga lebih rendah."]
    else:
        lines += ["  1. Promo cashback untuk aktivasi penggunaan pertama.",
                  "  2. Bonus reward transaksi pertama dalam 90 hari.",
                  "  3. Penutupan kartu jika tidak ada aktivitas 12 bulan."]
    lines.append("")

lines += ["\n"+"="*65,"ANALISIS NOISE DBSCAN","="*65]
if DB_NOISE > 0:
    nm = db_labels==-1
    np_profile = df_imp[nm][['BALANCE','PURCHASES','CASH_ADVANCE','CREDIT_LIMIT','PRC_FULL_PAYMENT']].mean()
    all_profile = df_imp[['BALANCE','PURCHASES','CASH_ADVANCE','CREDIT_LIMIT','PRC_FULL_PAYMENT']].mean()
    lines.append(f"\nJumlah noise: {DB_NOISE} ({DB_NOISE_PCT:.1f}% dari {N_ROWS})")
    lines.append("\nRata-rata fitur NOISE vs SEMUA NASABAH:")
    for f in np_profile.index:
        lines.append(f"  {f:<30}: Noise={np_profile[f]:>10.2f}  | Semua={all_profile[f]:>10.2f}")
    lines += ["","Interpretasi:",
              "  Nasabah noise adalah outlier statistik — pola perilaku mereka begitu",
              "  ekstrem sehingga DBSCAN tidak memasukkan mereka ke cluster manapun.",
              "  Mereka bisa jadi: nasabah VIP dengan limit sangat tinggi, atau nasabah",
              "  bermasalah dengan cash advance sangat besar, atau korporat dengan pola",
              "  berbeda dari retail. Bank sebaiknya menginvestigasi mereka secara manual.",
              "  DBSCAN adalah satu-satunya algoritma di proyek ini yang bisa mendeteksi",
              "  outlier ini secara otomatis — K-Means memaksanya masuk ke suatu cluster."]

with open(os.path.join(REP,'interpretasi_bisnis.txt'),'w',encoding='utf-8') as f:
    f.write('\n'.join(lines))
print("interpretasi_bisnis.txt saved")

# ── tabel_perbandingan.txt ─────────────────────────────────────
print("Writing tabel_perbandingan.txt...")
tbl = [
    "PERBANDINGAN K-MEANS vs DBSCAN",
    f"Dataset: Credit Card Customer Segmentation (N = {N_ROWS} nasabah)",
    "─"*79,
    f"{'Kriteria':<28} │ {'K-Means':<22} │ {'DBSCAN':<22}",
    "─"*79,
]
rows_t = [
    ("Metode penentuan K",      "Elbow + Silhouette",        "Otomatis (densitas)"),
    ("Parameter",               f"K = {K_OPT}",              f"eps={EPS_FINAL:.3f}, min_s={MIN_SAMPLES}"),
    ("Jumlah cluster",          str(K_OPT),                  str(DB_CLUSTERS)),
    ("Noise/Outlier",           "0 (tidak ada)",             f"{DB_NOISE} ({DB_NOISE_PCT:.1f}%)"),
    ("Silhouette Score ↑",      f"{KM_SIL:.4f}",            f"{DB_SIL:.4f}"),
    ("Davies-Bouldin ↓",        f"{KM_DB:.4f}",             "N/A"),
    ("Inertia",                 f"{KM_INERTIA:.0f}",        "N/A"),
    ("Bentuk cluster",          "Globular/bulat",            "Arbitrer (bebas)"),
    ("Keunggulan",              "Sederhana, cepat",         "Deteksi outlier otomatis"),
    ("Kelemahan",               "K harus ditentukan",       "Sensitif eps"),
    ("Cocok untuk",             "Segmentasi marketing",     "Deteksi anomali/fraud"),
]
for c,k,d in rows_t:
    tbl.append(f"{c:<28} │ {k:<22} │ {d:<22}")
tbl.append("─"*79)
tbl.append(f"PEMENANG (Silhouette): {pemenang}")
tbl.append("─"*79)
tbl.append("")
tbl.append("KESIMPULAN:")
if pemenang == "K-Means":
    tbl += [f"  K-Means unggul Silhouette ({KM_SIL:.4f} vs {DB_SIL:.4f}) pada dataset ini.",
            f"  Membagi {N_ROWS} nasabah ke {K_OPT} segmen jelas dan mudah diinterpretasikan.",
            f"  DBSCAN tetap bernilai: mendeteksi {DB_NOISE} outlier ({DB_NOISE_PCT:.1f}%) secara otomatis.",
            f"  Rekomendasi: K-Means untuk segmentasi marketing, DBSCAN untuk deteksi fraud."]
else:
    tbl += [f"  DBSCAN unggul Silhouette ({DB_SIL:.4f} vs {KM_SIL:.4f}) pada dataset ini.",
            f"  Selain itu mendeteksi {DB_NOISE} outlier ({DB_NOISE_PCT:.1f}%) secara otomatis.",
            f"  K-Means tetap berguna untuk segmentasi bisnis yang mudah dijelaskan ke stakeholder.",
            f"  Rekomendasi: DBSCAN untuk anomali, K-Means untuk kampanye marketing."]

with open(os.path.join(REP,'tabel_perbandingan.txt'),'w',encoding='utf-8') as f:
    f.write('\n'.join(tbl))
print("tabel_perbandingan.txt saved")

# ── Update state ───────────────────────────────────────────────
S['pemenang'] = pemenang
with open(os.path.join(BASE,'_state.json'),'w') as f: json.dump(S, f)

print("\nSTAGE 4 COMPLETE")
print(json.dumps(meta, indent=2, ensure_ascii=False))
