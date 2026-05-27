"""
Phase 3+4 v3: 튜닝된 클러스터링
Cluster 5 거대 클러스터 문제 해결
"""

import os
import numpy as np
import pandas as pd
import json
import hdbscan
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_PATH = "/home/hs/oer-catalyst-project/output/sentences_embeddings.npy"
SENTENCES_PATH = "/home/hs/oer-catalyst-project/output/sentences.csv"
BASE = "/home/hs/oer-catalyst-project/output"

def main():
    embeddings = np.load(EMBEDDINGS_PATH)
    df = pd.read_csv(SENTENCES_PATH)
    sentences = df['sentence'].tolist()
    print(f"Embeddings: {embeddings.shape}, Sentences: {len(sentences)}")

    # 여러 파라미터 조합 실험
    configs = [
        {"name": "v3a", "n_neighbors": 15, "min_dist": 0.0, "min_cluster_size": 50, "min_samples": 5},
        {"name": "v3b", "n_neighbors": 20, "min_dist": 0.1, "min_cluster_size": 50, "min_samples": 8},
        {"name": "v3c", "n_neighbors": 10, "min_dist": 0.0, "min_cluster_size": 40, "min_samples": 5},
    ]

    for cfg in configs:
        print(f"\n{'#'*60}")
        print(f"Config: {cfg['name']} (nn={cfg['n_neighbors']}, min_cs={cfg['min_cluster_size']})")
        print(f"{'#'*60}")

        # UMAP 15D
        reducer_15d = umap.UMAP(
            n_components=15, n_neighbors=cfg['n_neighbors'],
            min_dist=cfg['min_dist'], metric='cosine', random_state=42,
        )
        umap_15d = reducer_15d.fit_transform(embeddings)

        # HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg['min_cluster_size'],
            min_samples=cfg['min_samples'],
            metric='euclidean', cluster_selection_method='eom',
        )
        labels = clusterer.fit_predict(umap_15d)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        # 가장 큰 클러스터 비율
        sizes = [(labels == c).sum() for c in range(n_clusters)]
        max_size = max(sizes) if sizes else 0
        
        print(f"  Clusters: {n_clusters}, Noise: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        print(f"  Max cluster: {max_size} ({max_size/len(labels)*100:.1f}%)")
        print(f"  Size distribution: {sorted(sizes, reverse=True)[:10]}")

        # 결과가 좋으면 상세 출력 + 저장
        max_ratio = max_size / len(labels)
        if max_ratio < 0.5 and n_clusters >= 15 and n_clusters <= 50:
            print(f"\n  >>> GOOD CONFIG: {cfg['name']} <<<")
            
            # UMAP 2D
            reducer_2d = umap.UMAP(
                n_components=2, n_neighbors=cfg['n_neighbors'],
                min_dist=0.1, metric='cosine', random_state=42,
            )
            umap_2d = reducer_2d.fit_transform(embeddings)
            
            # 노이즈 재배정
            cluster_ids = sorted(set(labels) - {-1})
            centroids = {c: embeddings[labels==c].mean(axis=0) for c in cluster_ids}
            for idx in np.where(labels == -1)[0]:
                emb = embeddings[idx].reshape(1, -1)
                best_c, best_sim = -1, -1
                for c, cent in centroids.items():
                    sim = cosine_similarity(emb, cent.reshape(1, -1))[0, 0]
                    if sim > best_sim:
                        best_sim = sim
                        best_c = c
                labels[idx] = best_c

            np.save(f"{BASE}/clustering_labels_{cfg['name']}.npy", labels)
            np.save(f"{BASE}/umap_2d_{cfg['name']}.npy", umap_2d)

            # 해석
            summaries = []
            for c in cluster_ids:
                mask = labels == c
                cluster_sents = [s for s, m in zip(sentences, mask) if m]
                cluster_embs = embeddings[mask]
                
                try:
                    vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1,2), token_pattern=r'[A-Za-z0-9]+[A-Za-z0-9\.\-]*')
                    tfidf = vectorizer.fit_transform(cluster_sents)
                    features = vectorizer.get_feature_names_out()
                    mean_tfidf = tfidf.mean(axis=0).A1
                    top_kw = [features[i] for i in mean_tfidf.argsort()[-15:][::-1]]
                except:
                    top_kw = []
                
                centroid = cluster_embs.mean(axis=0, keepdims=True)
                sims = cosine_similarity(cluster_embs, centroid).flatten()
                top_idx = sims.argsort()[-5:][::-1]
                rep = [cluster_sents[i] for i in top_idx]
                
                summaries.append({
                    'cluster_id': int(c),
                    'size': int(mask.sum()),
                    'top_keywords': top_kw,
                    'representative_sentences': rep,
                })
                
                print(f"\n  Cluster {c} ({mask.sum()} sentences)")
                print(f"    Keywords: {', '.join(top_kw[:8])}")
                for j, s in enumerate(rep[:2]):
                    print(f"    Rep {j+1}: {s[:120]}...")

            with open(f"{BASE}/clusters_summary_{cfg['name']}.json", 'w') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
            print(f"\n  Saved: clusters_summary_{cfg['name']}.json")
            break  # 좋은 결과 찾으면 중단

    print("\nDone!")


if __name__ == '__main__':
    main()
