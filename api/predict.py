"""
API Flask untuk prediksi cluster nasabah kartu kredit.
Project 3 Machine Learning - Kelompok B
"""
import os, json, traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

# Lazy-load models
_models = {}

def load_models():
    if not _models:
        _models['imputer']  = joblib.load(os.path.join(MODEL_DIR, 'imputer.pkl'))
        _models['scaler']   = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        _models['pca']      = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
        _models['kmeans']   = joblib.load(os.path.join(MODEL_DIR, 'kmeans_final.pkl'))
        _models['dbscan']   = joblib.load(os.path.join(MODEL_DIR, 'dbscan_final.pkl'))
        with open(os.path.join(MODEL_DIR, 'metadata.json'), encoding='utf-8') as f:
            _models['meta'] = json.load(f)
    return _models

FEATURES = [
    'BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY',
    'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY',
    'CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX',
    'CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE'
]

CLUSTER_LABELS = {}  # akan diisi dari metadata jika tersedia

def get_km_label_name(cluster_id, meta):
    names = meta.get('kmeans', {}).get('cluster_names', {})
    return names.get(str(cluster_id), f"Cluster {cluster_id}")


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'ML Clustering API - Kelompok B'})


@app.route('/api/stats', methods=['GET'])
def stats():
    try:
        m = load_models()['meta']
        return jsonify(m)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        m = load_models()
        data = request.get_json(force=True)

        # Bangun vektor fitur dengan default 0 untuk fitur yang tidak dikirim
        input_vec = []
        for feat in FEATURES:
            val = data.get(feat, 0)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
            input_vec.append(val)

        X_raw = np.array(input_vec).reshape(1, -1)
        X_imp = m['imputer'].transform(X_raw)
        X_sc  = m['scaler'].transform(X_imp)
        X_pca = m['pca'].transform(X_sc)

        # K-Means prediction
        km_label = int(m['kmeans'].predict(X_sc)[0])

        # DBSCAN: nearest-neighbor ke cluster centers yang sudah diketahui
        # Implementasi: cari label cluster yang paling banyak di k-nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        # Load training data cluster labels dari hasil_clustering.csv
        try:
            df_hasil = pd.read_csv(os.path.join(DATA_DIR, 'hasil_clustering.csv'))
            X_train_scaled = m['scaler'].transform(
                m['imputer'].transform(df_hasil[FEATURES].values)
            )
            db_train_labels = df_hasil['DBSCAN_Label'].values
            nbrs = NearestNeighbors(n_neighbors=5).fit(X_train_scaled)
            _, indices = nbrs.kneighbors(X_sc)
            neighbor_labels = db_train_labels[indices[0]]
            # Majority vote, exclude -1 (noise)
            valid_labels = neighbor_labels[neighbor_labels != -1]
            if len(valid_labels) > 0:
                from collections import Counter
                db_label = int(Counter(valid_labels).most_common(1)[0][0])
                is_noise = False
            else:
                db_label = -1
                is_noise = True
        except Exception:
            db_label = -1
            is_noise = True

        meta = m['meta']
        km_name = get_km_label_name(km_label, meta)

        return jsonify({
            'kmeans': {
                'cluster': km_label,
                'nama': km_name,
                'total_clusters': meta['kmeans']['k_optimal']
            },
            'dbscan': {
                'cluster': db_label,
                'is_noise': is_noise,
                'label_text': 'Outlier/Noise — Perilaku sangat tidak biasa' if is_noise else f'Cluster {db_label}'
            },
            'pca_coordinates': {
                'PC1': float(X_pca[0, 0]),
                'PC2': float(X_pca[0, 1])
            },
            'input_fitur': dict(zip(FEATURES, [float(v) for v in X_imp[0]]))
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/pca_data', methods=['GET'])
def pca_data():
    try:
        sample_size = int(request.args.get('n', 2000))
        df = pd.read_csv(os.path.join(DATA_DIR, 'hasil_clustering.csv'))
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        result = df[['PC1', 'PC2', 'KMeans_Label', 'DBSCAN_Label']].to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
