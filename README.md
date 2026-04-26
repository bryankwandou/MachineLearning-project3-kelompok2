# Project 3 Machine Learning — Kelompok B
## Unsupervised Learning: K-Means Clustering vs DBSCAN

**Demo Live:** https://machinelearning-kelompok2.vercel.app/  
**Mata Kuliah:** Machine dan Deep Learning (473TI4)  
**Dosen:** Dr. Ir. Norbertus Tri Suswanto Saptadi, S.Kom., MT., MM., IPM.  
**Universitas:** Universitas Atma Jaya Makassar  
**Kelompok B:** Brad | Annaliese | Vincentius | Mike

---

## Dataset
**Nama:** Credit Card Customer Segmentation  
**Sumber:** Kaggle — https://www.kaggle.com/datasets/arjunbhasin2013/ccdata  
**Jumlah Data:** 8.950 nasabah  
**Jumlah Fitur:** 17 fitur perilaku finansial  
**Tujuan:** Mengelompokkan nasabah kartu kredit berdasarkan pola perilaku penggunaan kartu

## Cara Download Dataset
1. Buka https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
2. Login ke akun Kaggle, klik Download
3. Letakkan file `CC GENERAL.csv` di folder `data/`

## Hasil Evaluasi Model

| Metrik | K-Means | DBSCAN |
|---|---|---|
| Jumlah Cluster | 3 | 2 |
| Silhouette Score | 0.2466 | 0.7208 |
| Davies-Bouldin | 1.5920 | N/A |
| Data Noise | 0% | 0.55% (49 nasabah) |
| PCA Variance PC1+PC2 | 47.61% | 47.61% |

**Pemenang: DBSCAN** (Silhouette 0.7208 vs 0.2466)

## Cara Menjalankan

```bash
pip install -r requirements.txt
# Jalankan notebook
jupyter notebook notebooks/project3_clustering_kelompok_b.ipynb
# Atau jalankan pipeline langsung
python stage1_prep.py
python stage2_kmeans.py
python stage3_dbscan.py
python stage4_reports.py
```

## Struktur Folder

```
MachineLearning-project3-kelompok2/
├── notebooks/      # Jupyter notebook lengkap
├── data/           # preprocessed.csv, hasil_clustering.csv, pca_result.csv
├── models/         # kmeans.pkl, dbscan.pkl, scaler.pkl, pca.pkl, metadata.json
├── api/            # Flask API untuk prediksi
├── frontend/       # Dashboard web interaktif
├── reports/        # 11 grafik + interpretasi bisnis + tabel perbandingan
├── requirements.txt
└── vercel.json
```

## Teknologi
Python 3.12 | scikit-learn | pandas | numpy | matplotlib | seaborn | Flask | Vercel | GitHub
