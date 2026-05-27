"""
Phase 1-6: 새 논문 309편 분석 파이프라인
Abstract 기반 문장 추출 → 임베딩 → 클러스터링 → 해석 → 시각화
"""

import json
import pandas as pd
import numpy as np
import re
import os
from collections import Counter

OUTPUT = "/home/hs/oer-catalyst-project/output/papers"
os.makedirs(OUTPUT, exist_ok=True)

# ============================================================
# Phase 1: 문장 추출
# ============================================================
def extract_sentences():
    print("=" * 60)
    print("  Phase 1: 문장 추출")
    print("=" * 60)

    data = json.load(open(os.path.join(OUTPUT, "oer_papers_db.json")))

    sentences = []
    for paper in data:
        pid = paper["paper_id"]
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        year = paper.get("year")
        citations = paper.get("citations", 0)
        doi = paper.get("doi", "")
        journal = paper.get("journal", "")
        authors = paper.get("authors", "")

        if not abstract or len(abstract.split()) < 20:
            continue

        # Split abstract into sentences
        # Also prepend title as a sentence for context
        raw_sents = re.split(r'(?<=[.!?])\s+', abstract.strip())

        for sent in raw_sents:
            sent = sent.strip()
            if len(sent.split()) < 5:  # Skip very short fragments
                continue
            sentences.append({
                "source_id": pid,
                "title": title,
                "sentence": sent,
                "year": year,
                "citations": citations,
                "doi": doi,
                "journal": journal,
                "authors": authors,
                "is_title": False,
            })

    df = pd.DataFrame(sentences)
    out_path = os.path.join(OUTPUT, "sentences_extracted.csv")
    df.to_csv(out_path, index=False)

    print(f"  추출: {len(df)}문장 / {len(data)}논문")
    print(f"  평균 문장/논문: {len(df)/len(data):.1f}")
    print(f"  저장: {out_path}")
    return df


# ============================================================
# Phase 2: 임베딩
# ============================================================
def embed_sentences(df):
    print(f"\n{'='*60}")
    print("  Phase 2: 임베딩 (bge-small-en-v1.5)")
    print(f"{'='*60}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    texts = df["sentence"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    emb_path = os.path.join(OUTPUT, "embeddings.npy")
    np.save(emb_path, embeddings)

    print(f"  임베딩: {embeddings.shape}")
    print(f"  저장: {emb_path}")
    return embeddings


# ============================================================
# Phase 3+4: UMAP + HDBSCAN 클러스터링
# ============================================================
def cluster_sentences(embeddings):
    print(f"\n{'='*60}")
    print("  Phase 3+4: UMAP + HDBSCAN 클러스터링")
    print(f"{'='*60}")

    import umap
    import hdbscan

    # UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    umap_emb = reducer.fit_transform(embeddings)

    # 2D for visualization
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    umap_2d = reducer_2d.fit_transform(embeddings)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(umap_emb)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = sum(1 for l in labels if l == -1)
    noise_pct = noise / len(labels) * 100

    print(f"  클러스터: {n_clusters}개")
    print(f"  노이즈: {noise} ({noise_pct:.1f}%)")

    # Cluster sizes
    sizes = Counter(labels)
    print(f"  클러스터 크기 (Top 10):")
    for label, count in sizes.most_common(10):
        name = f"Cluster {label}" if label >= 0 else "Noise"
        print(f"    {name}: {count}")

    return umap_2d, labels, n_clusters


# ============================================================
# Phase 5: 클러스터 해석 (TF-IDF + 대표 문장)
# ============================================================
def interpret_clusters(df, labels, n_clusters):
    print(f"\n{'='*60}")
    print("  Phase 5: 클러스터 해석")
    print(f"{'='*60}")

    from sklearn.feature_extraction.text import TfidfVectorizer

    df = df.copy()
    df["cluster"] = labels

    # TF-IDF per cluster
    cluster_info = []
    for c in range(n_clusters):
        mask = df["cluster"] == c
        cluster_sents = df[mask]["sentence"].tolist()
        cluster_size = mask.sum()

        if cluster_size < 3:
            continue

        # TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=100, stop_words="english",
            ngram_range=(1, 2), max_df=0.8
        )
        try:
            tfidf = vectorizer.fit_transform(cluster_sents)
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = tfidf.mean(axis=0).A1
            top_keywords = [
                feature_names[i] for i in mean_tfidf.argsort()[-10:][::-1]
            ]
        except Exception:
            top_keywords = []

        # Top cited representative sentences
        cluster_df = df[mask].sort_values("citations", ascending=False)
        top_sents = cluster_df.head(3)[["sentence", "title", "citations"]].to_dict("records")

        # Year distribution
        years = df[mask]["year"].dropna()
        year_range = f"{int(years.min())}-{int(years.max())}" if len(years) > 0 else "N/A"

        # Average citations
        avg_cites = df[mask]["citations"].mean()

        info = {
            "cluster": c,
            "size": cluster_size,
            "keywords": ", ".join(top_keywords[:7]),
            "top_keywords_list": top_keywords,
            "representative_sentences": top_sents,
            "year_range": year_range,
            "avg_citations": round(avg_cites, 0),
        }
        cluster_info.append(info)

        print(f"\n  Cluster {c} ({cluster_size}문장, avg {avg_cites:.0f} cites):")
        print(f"    키워드: {', '.join(top_keywords[:7])}")
        print(f"    연도: {year_range}")
        for s in top_sents[:2]:
            print(f"    → {s['sentence'][:80]}...")

    # Save
    info_path = os.path.join(OUTPUT, "cluster_interpretation.json")
    # Convert for JSON serialization
    json_safe = []
    for info in cluster_info:
        js = {}
        for k, v in info.items():
            if isinstance(v, (np.integer,)):
                js[k] = int(v)
            elif isinstance(v, (np.floating,)):
                js[k] = float(v)
            elif isinstance(v, np.ndarray):
                js[k] = v.tolist()
            else:
                js[k] = v
        json_safe.append(js)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, ensure_ascii=False, indent=2)

    print(f"\n  저장: {info_path}")
    return df, cluster_info


# ============================================================
# Phase 6: 시각화
# ============================================================
def visualize(df, umap_2d, labels, cluster_info):
    print(f"\n{'='*60}")
    print("  Phase 6: 시각화")
    print(f"{'='*60}")

    import plotly.express as px

    df["x"] = umap_2d[:, 0]
    df["y"] = umap_2d[:, 1]
    df["cluster_label"] = df["cluster"].apply(
        lambda x: f"C{x}" if x >= 0 else "Noise"
    )

    # Scatter plot
    fig = px.scatter(
        df[df["cluster"] >= 0],
        x="x", y="y",
        color="cluster_label",
        hover_data=["sentence", "title", "citations"],
        title=f"OER 논문 클러스터링 ({len(df[df['cluster']>=0])}문장, {len(cluster_info)}클러스터)",
        width=1200, height=800,
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=4))
    html_path = os.path.join(OUTPUT, "cluster_scatter.html")
    fig.write_html(html_path)
    print(f"  산점도: {html_path}")

    # Bar chart - cluster sizes
    sizes_df = pd.DataFrame([
        {"cluster": f"C{c['cluster']}", "size": c["size"], "keywords": c["keywords"]}
        for c in cluster_info
    ])
    fig2 = px.bar(
        sizes_df, x="cluster", y="size",
        hover_data=["keywords"],
        title="클러스터별 문장 수",
    )
    bar_path = os.path.join(OUTPUT, "cluster_sizes.html")
    fig2.write_html(bar_path)
    print(f"  바차트: {bar_path}")

    # Save final dataframe
    final_path = os.path.join(OUTPUT, "sentences_clustered.csv")
    df.to_csv(final_path, index=False)
    print(f"  최종 데이터: {final_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  OER 논문 분석 파이프라인")
    print(f"{'='*60}\n")

    # Phase 1
    df = extract_sentences()

    # Phase 2
    embeddings = embed_sentences(df)

    # Phase 3+4
    umap_2d, labels, n_clusters = cluster_sentences(embeddings)

    # Phase 5
    df, cluster_info = interpret_clusters(df, labels, n_clusters)

    # Phase 6
    visualize(df, umap_2d, labels, cluster_info)

    print(f"\n{'='*60}")
    print("  분석 완료!")
    print(f"  클러스터: {n_clusters}개")
    print(f"  문장: {len(df)}개")
    print(f"{'='*60}")
