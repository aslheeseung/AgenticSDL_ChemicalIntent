"""
Phase 3+4: UMAP Dimensionality Reduction + HDBSCAN Clustering
임베딩 벡터를 UMAP으로 축소하고 HDBSCAN으로 클러스터링
"""

import os
import numpy as np
import pandas as pd
import json
import hdbscan
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 설정
# ============================================================
EMBEDDINGS_PATH = "/home/hs/oer-catalyst-project/output/sentences_embeddings.npy"
SENTENCES_PATH = "/home/hs/oer-catalyst-project/output/sentences.csv"
UMAP_2D_PATH = "/home/hs/oer-catalyst-project/output/umap_2d.npy"
UMAP_15D_PATH = "/home/hs/oer-catalyst-project/output/umap_15d.npy"
LABELS_PATH = "/home/hs/oer-catalyst-project/output/clustering_labels.npy"
SUMMARY_PATH = "/home/hs/oer-catalyst-project/output/clusters_summary.json"

# UMAP 파라미터
UMAP_15D_PARAMS = {
    'n_components': 15,
    'n_neighbors': 15,
    'min_dist': 0.0,
    'metric': 'cosine',
    'random_state': 42,
}

UMAP_2D_PARAMS = {
    'n_components': 2,
    'n_neighbors': 15,
    'min_dist': 0.1,
    'metric': 'cosine',
    'random_state': 42,
}

# HDBSCAN 파라미터
HDBSCAN_PARAMS = {
    'min_cluster_size': 30,
    'min_samples': 5,
    'metric': 'euclidean',
    'cluster_selection_method': 'eom',
}


# ============================================================
# Phase 3: UMAP 차원 축소
# ============================================================
def run_umap():
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings shape: {embeddings.shape}")

    # 384D → 15D (클러스터링용)
    print("\nUMAP: 384D → 15D (for clustering)...")
    reducer_15d = umap.UMAP(**UMAP_15D_PARAMS)
    umap_15d = reducer_15d.fit_transform(embeddings)
    np.save(UMAP_15D_PATH, umap_15d)
    print(f"Saved: {umap_15d.shape}")

    # 384D → 2D (시각화용)
    print("\nUMAP: 384D → 2D (for visualization)...")
    reducer_2d = umap.UMAP(**UMAP_2D_PARAMS)
    umap_2d = reducer_2d.fit_transform(embeddings)
    np.save(UMAP_2D_PATH, umap_2d)
    print(f"Saved: {umap_2d.shape}")

    return umap_15d, umap_2d


# ============================================================
# Phase 4: HDBSCAN 클러스터링
# ============================================================
def run_hdbscan(umap_15d):
    print("\nHDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    labels = clusterer.fit_predict(umap_15d)

    np.save(LABELS_PATH, labels)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print(f"\nCluster sizes:")
    for c in sorted(set(labels)):
        size = (labels == c).sum()
        tag = " (noise)" if c == -1 else ""
        print(f"  Cluster {c}: {size}{tag}")

    return labels, n_clusters


# ============================================================
# Phase 5: 클러스터 해석
# ============================================================
def interpret_clusters(labels, n_clusters):
    print("\nLoading sentences...")
    df = pd.read_csv(SENTENCES_PATH)
    sentences = df['sentence'].tolist()
    embeddings = np.load(EMBEDDINGS_PATH)

    summaries = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_sentences = [s for s, m in zip(sentences, mask) if m]
        cluster_embeddings = embeddings[mask]
        
        if len(cluster_sentences) == 0:
            continue

        # 1. TF-IDF 키워드 추출
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                token_pattern=r'[A-Za-z0-9]+[A-Za-z0-9\-\.]*',
            )
            tfidf_matrix = vectorizer.fit_transform(cluster_sentences)
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-15:][::-1]
            top_keywords = [feature_names[i] for i in top_indices]
        except:
            top_keywords = []

        # 2. 클러스터 중심에서 가장 가까운 대표 문장 5개
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embeddings, centroid).flatten()
        top_sent_indices = sims.argsort()[-5:][::-1]
        representative = [cluster_sentences[i] for i in top_sent_indices]

        summaries.append({
            'cluster_id': int(cluster_id),
            'size': int(mask.sum()),
            'top_keywords': top_keywords,
            'representative_sentences': representative,
            'suggested_intent': None,
            'confirmed_intent': None,
        })

        print(f"\n{'='*60}")
        print(f"Cluster {cluster_id} ({mask.sum()} sentences)")
        print(f"Keywords: {', '.join(top_keywords[:10])}")
        print(f"Representative sentences:")
        for j, s in enumerate(representative[:3]):
            print(f"  {j+1}. {s[:120]}...")

    # 노이즈 정보 추가
    noise_mask = labels == -1
    summaries.append({
        'cluster_id': -1,
        'size': int(noise_mask.sum()),
        'top_keywords': [],
        'representative_sentences': [],
        'suggested_intent': 'noise',
        'confirmed_intent': None,
    })

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary to {SUMMARY_PATH}")

    return summaries


# ============================================================
# 메인
# ============================================================
if __name__ == '__main__':
    umap_15d, umap_2d = run_umap()
    labels, n_clusters = run_hdbscan(umap_15d)
    summaries = interpret_clusters(labels, n_clusters)
    print("\nDone!")
