"""
Phase 3+4 v2: 2단계 클러스터링
  Step A: HDBSCAN (min_cluster_size 증가) → 노이즈 제거 + 1차 클러스터링
  Step C: Agglomerative Clustering → 유사 클러스터 병합
"""

import os
import numpy as np
import pandas as pd
import json
import hdbscan
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 설정
# ============================================================
EMBEDDINGS_PATH = "/home/hs/oer-catalyst-project/output/sentences_embeddings.npy"
SENTENCES_PATH = "/home/hs/oer-catalyst-project/output/sentences.csv"
UMAP_15D_PATH = "/home/hs/oer-catalyst-project/output/umap_15d_v2.npy"
UMAP_2D_PATH = "/home/hs/oer-catalyst-project/output/umap_2d_v2.npy"
LABELS_PATH = "/home/hs/oer-catalyst-project/output/clustering_labels_v2.npy"
MERGED_LABELS_PATH = "/home/hs/oer-catalyst-project/output/clustering_labels_merged.npy"
SUMMARY_PATH = "/home/hs/oer-catalyst-project/output/clusters_summary_v2.json"
DENDROGRAM_PATH = "/home/hs/oer-catalyst-project/output/dendrogram.png"

# Step A: HDBSCAN (파라미터 완화)
HDBSCAN_PARAMS = {
    'min_cluster_size': 80,      # 30 → 80 (클러스터 최소 크기 증가)
    'min_samples': 10,           # 5 → 10 (더 보수적으로)
    'metric': 'euclidean',
    'cluster_selection_method': 'eom',
}

# Step C: Agglomerative (최종 클러스터 수)
# 덴드로그램 보고 결정하지만 기본값 설정
N_FINAL_CLUSTERS = 25  # 75 → 25개로 압축

UMAP_15D_PARAMS = {
    'n_components': 15,
    'n_neighbors': 25,       # 15 → 25 (글로벌 구조 강화)
    'min_dist': 0.0,
    'metric': 'cosine',
    'random_state': 42,
}

UMAP_2D_PARAMS = {
    'n_components': 2,
    'n_neighbors': 25,
    'min_dist': 0.1,
    'metric': 'cosine',
    'random_state': 42,
}


def run_step_a():
    """Step A: UMAP + HDBSCAN (완화된 파라미터)"""
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings: {embeddings.shape}")

    # UMAP 15D
    print("\n[Step A] UMAP 384D → 15D...")
    reducer_15d = umap.UMAP(**UMAP_15D_PARAMS)
    umap_15d = reducer_15d.fit_transform(embeddings)
    np.save(UMAP_15D_PATH, umap_15d)
    print(f"  Saved: {umap_15d.shape}")

    # UMAP 2D
    print("\n[Step A] UMAP 384D → 2D...")
    reducer_2d = umap.UMAP(**UMAP_2D_PARAMS)
    umap_2d = reducer_2d.fit_transform(embeddings)
    np.save(UMAP_2D_PATH, umap_2d)
    print(f"  Saved: {umap_2d.shape}")

    # HDBSCAN
    print("\n[Step A] HDBSCAN (min_cluster_size=80, min_samples=10)...")
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    labels = clusterer.fit_predict(umap_15d)
    np.save(LABELS_PATH, labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    for c in sorted(set(labels)):
        size = (labels == c).sum()
        tag = " (noise)" if c == -1 else ""
        print(f"    Cluster {c}: {size}{tag}")

    return labels, n_clusters, embeddings


def run_step_c(labels_a, n_clusters_a, embeddings):
    """Step C: Agglomerative Clustering으로 클러스터 병합"""
    print(f"\n[Step C] Agglomerative Clustering on {n_clusters_a} cluster centroids...")

    # 각 HDBSCAN 클러스터의 중심점 계산
    cluster_ids = sorted([c for c in set(labels_a) if c != -1])
    centroids = []
    for c in cluster_ids:
        mask = labels_a == c
        centroids.append(embeddings[mask].mean(axis=0))
    centroids = np.array(centroids)
    print(f"  Centroids: {centroids.shape}")

    # 덴드로그램용 linkage matrix
    print("  Computing linkage matrix...")
    Z = linkage(centroids, method='average', metric='cosine')

    # 덴드로그램 저장
    fig, ax = plt.subplots(figsize=(16, 8))
    labels_list = [f"C{c} ({(labels_a==c).sum()})" for c in cluster_ids]
    dendrogram(Z, labels=labels_list, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title("Dendrogram: HDBSCAN Cluster Centroids (cosine, average linkage)")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    fig.savefig(DENDROGRAM_PATH, dpi=150)
    plt.close()
    print(f"  Saved dendrogram: {DENDROGRAM_PATH}")

    # Agglomerative Clustering
    agg = AgglomerativeClustering(
        n_clusters=N_FINAL_CLUSTERS,
        metric='cosine',
        linkage='average',
    )
    merge_labels = agg.fit_predict(centroids)
    print(f"\n  Merged into {N_FINAL_CLUSTERS} super-clusters")

    # 매핑: 원래 HDBSCAN label → merged label
    hdb_to_merged = {}
    for i, c in enumerate(cluster_ids):
        hdb_to_merged[c] = int(merge_labels[i])

    # 전체 문장에 merged label 할당
    merged_labels = np.full(len(labels_a), -1, dtype=int)
    for i, lb in enumerate(labels_a):
        if lb != -1:
            merged_labels[i] = hdb_to_merged[lb]

    np.save(MERGED_LABELS_PATH, merged_labels)

    # 노이즈 재활당 시도: 노이즈를 가장 가까운 클러스터 중심에 배정
    noise_mask = labels_a == -1
    noise_indices = np.where(noise_mask)[0]
    print(f"\n  Reassigning {len(noise_indices)} noise points to nearest cluster...")

    # merged cluster 중심점
    merged_centroids = {}
    for mc in range(N_FINAL_CLUSTERS):
        mask = merged_labels == mc
        if mask.sum() > 0:
            merged_centroids[mc] = embeddings[mask].mean(axis=0)

    reassigned = merged_labels.copy()
    for idx in noise_indices:
        emb = embeddings[idx].reshape(1, -1)
        best_cluster = -1
        best_sim = -1
        for mc, cent in merged_centroids.items():
            sim = cosine_similarity(emb, cent.reshape(1, -1))[0, 0]
            if sim > best_sim:
                best_sim = sim
                best_cluster = mc
        reassigned[idx] = best_cluster

    print(f"  Reassigned. Noise remaining: {(reassigned == -1).sum()}")
    np.save(MERGED_LABELS_PATH, reassigned)

    return reassigned


def interpret_merged(merged_labels):
    """최종 클러스터 해석"""
    print("\n[Interpretation] Analyzing merged clusters...")
    df = pd.read_csv(SENTENCES_PATH)
    sentences = df['sentence'].tolist()
    embeddings = np.load(EMBEDDINGS_PATH)

    n_final = len(set(merged_labels)) - (1 if -1 in merged_labels else 0)
    summaries = []

    for cluster_id in range(n_final):
        mask = merged_labels == cluster_id
        cluster_sentences = [s for s, m in zip(sentences, mask) if m]
        cluster_embeddings = embeddings[mask]

        if len(cluster_sentences) == 0:
            continue

        # TF-IDF
        try:
            vectorizer = TfidfVectorizer(
                max_features=100, stop_words='english',
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

        # 대표 문장
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embeddings, centroid).flatten()
        top_idx = sims.argsort()[-5:][::-1]
        representative = [cluster_sentences[i] for i in top_idx]

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
        print(f"  Keywords: {', '.join(top_keywords[:10])}")
        print(f"  Representative:")
        for j, s in enumerate(representative[:3]):
            print(f"    {j+1}. {s[:130]}...")

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {SUMMARY_PATH}")

    return summaries


if __name__ == '__main__':
    labels_a, n_clusters_a, embeddings = run_step_a()
    merged_labels = run_step_c(labels_a, n_clusters_a, embeddings)
    summaries = interpret_merged(merged_labels)
    print("\nDone!")
