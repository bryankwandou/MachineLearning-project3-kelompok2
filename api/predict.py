"""
Flask API — Project 3 ML Kelompok B
Serves /api/* endpoints with hardcoded fallback if models unavailable.
"""
import os, json, datetime, traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins":"*","methods":["GET","POST","OPTIONS"],"allow_headers":["Content-Type"]}})

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_D = os.path.join(BASE, 'models')
DATA_D  = os.path.join(BASE, 'data')

METADATA_STATIS = {
    "nama_proyek": "Project 3 ML Kelompok B",
    "dataset": "Credit Card Customer Segmentation",
    "jumlah_baris": 8950, "jumlah_fitur": 17,
    "pca_variance_explained_pct": 47.61,
    "kmeans": {"k_optimal":3,"silhouette_score":0.2466,"davies_bouldin_score":1.5920,
               "inertia":111975.04,"distribusi_cluster":{"0":1275,"1":6114,"2":1561}},
    "dbscan": {"eps_optimal":5.4131,"min_samples":5,"jumlah_cluster":2,
               "jumlah_noise":49,"persen_noise":0.55,"silhouette_score":0.7208},
    "pemenang": "DBSCAN",
    "alasan_pemenang": "DBSCAN memiliki Silhouette Score 0.7208 jauh lebih tinggi dibanding K-Means 0.2466"
}

CLUSTER_NAMES = {0: "Nasabah Pasif", 1: "Nasabah Aktif Moderat", 2: "Nasabah Revolving/Berisiko"}

@app.after_request
def cors(r):
    r.headers.update({"Access-Control-Allow-Origin":"*",
                       "Access-Control-Allow-Methods":"GET,POST,OPTIONS",
                       "Access-Control-Allow-Headers":"Content-Type"})
    return r

@app.route('/api/health')
def health():
    files = {
        "metadata":     os.path.exists(os.path.join(MODEL_D,'metadata.json')),
        "pca_data":     os.path.exists(os.path.join(DATA_D,'pca_result.csv')),
        "kmeans_model": os.path.exists(os.path.join(MODEL_D,'kmeans_final.pkl')),
        "dbscan_model": os.path.exists(os.path.join(MODEL_D,'dbscan_final.pkl')),
    }
    return jsonify({"status":"ok","timestamp":datetime.datetime.utcnow().isoformat()+"Z","files_found":files})

@app.route('/api/stats')
def stats():
    try:
        path = os.path.join(MODEL_D,'metadata.json')
        if os.path.exists(path):
            with open(path,encoding='utf-8') as f:
                return jsonify(json.load(f))
    except Exception as e:
        print(f"ERROR /api/stats reading file: {e}")
    return jsonify(METADATA_STATIS)

@app.route('/api/predict', methods=['POST','OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    try:
        data = request.get_json(force=True) or {}
        bal  = float(data.get('BALANCE', data.get('balance', 0)))
        pur  = float(data.get('PURCHASES', data.get('purchases', 0)))
        ca   = float(data.get('CASH_ADVANCE', data.get('cash_advance', 0)))
        cl   = float(data.get('CREDIT_LIMIT', data.get('credit_limit', 5000)))
        pfp  = float(data.get('PRC_FULL_PAYMENT', data.get('prc_full_payment', 0)))
        ten  = float(data.get('TENURE', data.get('tenure', 12)))

        # Try real model prediction
        try:
            import joblib, numpy as np
            FEATURES = ['BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES',
                        'INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY',
                        'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY',
                        'CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX',
                        'CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']
            imp = joblib.load(os.path.join(MODEL_D,'imputer.pkl'))
            sc  = joblib.load(os.path.join(MODEL_D,'scaler.pkl'))
            pca = joblib.load(os.path.join(MODEL_D,'pca.pkl'))
            km  = joblib.load(os.path.join(MODEL_D,'kmeans_final.pkl'))
            full = {f: float(data.get(f,0)) for f in FEATURES}
            full['BALANCE']=bal; full['PURCHASES']=pur; full['CASH_ADVANCE']=ca
            full['CREDIT_LIMIT']=cl; full['PRC_FULL_PAYMENT']=pfp; full['TENURE']=ten
            X = np.array([full[f] for f in FEATURES]).reshape(1,-1)
            X_imp = imp.transform(X); X_sc = sc.transform(X_imp)
            X_pca = pca.transform(X_sc)
            km_lbl = int(km.predict(X_sc)[0])
            # DBSCAN kNN
            try:
                import pandas as pd
                from sklearn.neighbors import NearestNeighbors
                from collections import Counter
                db = joblib.load(os.path.join(MODEL_D,'dbscan_final.pkl'))
                df_h = pd.read_csv(os.path.join(DATA_D,'hasil_clustering.csv'))
                Xtr = sc.transform(imp.transform(df_h[FEATURES].values))
                nb  = NearestNeighbors(n_neighbors=7).fit(Xtr)
                _, idx = nb.kneighbors(X_sc)
                nbrs = df_h['DBSCAN_Label'].values[idx[0]]
                v = nbrs[nbrs != -1]
                db_lbl = int(Counter(v).most_common(1)[0][0]) if len(v) else -1
            except:
                db_lbl = 0
            return jsonify({
                "kmeans":{"cluster":km_lbl,"nama":CLUSTER_NAMES.get(km_lbl,f"Cluster {km_lbl}")},
                "dbscan":{"cluster":db_lbl,"is_noise":db_lbl==-1,
                          "label_text":"Outlier/Noise" if db_lbl==-1 else f"Cluster {db_lbl}"},
                "pca_coordinates":{"PC1":float(X_pca[0,0]),"PC2":float(X_pca[0,1])},
                "mode":"model"
            })
        except Exception as em:
            print(f"Model predict failed: {em}")

        # Rule-based fallback
        if ca > 3000:
            km_lbl, km_name = 2, "Nasabah Revolving/Berisiko"
        elif pur > 2000 and pfp > 0.25:
            km_lbl, km_name = 0, "Nasabah Pasif"
        else:
            km_lbl, km_name = 1, "Nasabah Aktif Moderat"
        db_lbl = 1 if pur > 500 else 0
        return jsonify({
            "kmeans":{"cluster":km_lbl,"nama":km_name},
            "dbscan":{"cluster":db_lbl,"is_noise":False,"label_text":f"Cluster {db_lbl}"},
            "pca_coordinates":{"PC1":0.0,"PC2":0.0},
            "mode":"simulasi"
        })
    except Exception as e:
        print(f"ERROR /api/predict: {traceback.format_exc()}")
        return jsonify({"error":str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
