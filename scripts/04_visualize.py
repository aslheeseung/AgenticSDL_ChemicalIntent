"""
Phase 6: Interactive Visualization
UMAP 2D scatter plot with hover text
"""

import os
import numpy as np
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 설정
# ============================================================
SENTENCES_PATH = "/home/hs/oer-catalyst-project/output/sentences.csv"
UMAP_2D_PATH = "/home/hs/oer-catalyst-project/output/umap_2d.npy"
LABELS_PATH = "/home/hs/oer-catalyst-project/output/clustering_labels.npy"
SUMMARY_PATH = "/home/hs/oer-catalyst-project/output/clusters_summary.json"
HTML_PATH = "/home/hs/oer-catalyst-project/output/interactive_plot.html"
PNG_PATH = "/home/hs/oer-catalyst-project/output/cluster_visualization.png"


def create_visualization():
    # 데이터 로드
    df = pd.read_csv(SENTENCES_PATH)
    umap_2d = np.load(UMAP_2D_PATH)
    labels = np.load(LABELS_PATH)

    with open(SUMMARY_PATH) as f:
        summaries = json.load(f)

    # 클러스터별 top keywords 매핑
    cluster_keywords = {}
    for s in summaries:
        kws = s.get('top_keywords', [])[:5]
        cluster_keywords[s['cluster_id']] = ', '.join(kws)

    # DataFrame 구성
    df['x'] = umap_2d[:, 0]
    df['y'] = umap_2d[:, 1]
    df['cluster'] = labels

    # 클러스터 라벨 (키워드 포함)
    df['cluster_label'] = df['cluster'].apply(
        lambda c: f"Cluster {c}: {cluster_keywords.get(c, 'noise')}" if c != -1 else "Noise"
    )

    # Hover text
    df['hover'] = df.apply(
        lambda r: f"[{r['cluster_label']}]<br>"
                  f"Paper: {r['paper_id']}<br>"
                  f"Section: {r['section']}<br>"
                  f"<b>{r['sentence'][:150]}</b>",
        axis=1
    )

    # 노이즈 분리
    df_noise = df[df['cluster'] == -1]
    df_clusters = df[df['cluster'] != -1]

    print(f"Clusters: {df_clusters['cluster'].nunique()}")
    print(f"Noise: {len(df_noise)}")
    print(f"Total: {len(df)}")

    # Plotly interactive plot
    fig = go.Figure()

    # 노이즈 포인트 (회색, 작고 투명)
    fig.add_trace(go.Scattergl(
        x=df_noise['x'],
        y=df_noise['y'],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.3),
        text=df_noise['hover'],
        hoverinfo='text',
        name='Noise',
    ))

    # 클러스터 포인트
    fig.add_trace(go.Scattergl(
        x=df_clusters['x'],
        y=df_clusters['y'],
        mode='markers',
        marker=dict(
            size=4,
            color=df_clusters['cluster'],
            colorscale=px.colors.qualitative.Alphabet + px.colors.qualitative.Set3,
            opacity=0.7,
            colorbar=dict(title="Cluster ID", tickfont=dict(size=10)),
        ),
        text=df_clusters['hover'],
        hoverinfo='text',
        name='Clusters',
    ))

    fig.update_layout(
        title=dict(
            text="OER Catalyst Synthesis - Sentence Clustering (UMAP 2D + HDBSCAN)",
            font=dict(size=16),
        ),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        width=1400,
        height=900,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.write_html(HTML_PATH)
    print(f"\nSaved interactive plot: {HTML_PATH}")
    print(f"File size: {os.path.getsize(HTML_PATH) / 1024:.1f} KB")

    # 정적 PNG (kaleido 필요시 스킵)
    try:
        fig.write_image(PNG_PATH, width=1400, height=900, scale=2)
        print(f"Saved static plot: {PNG_PATH}")
    except Exception as e:
        print(f"PNG export skipped (install kaleido): {e}")

    # 클러스터 크기 바 차트
    cluster_sizes = df[df['cluster'] != -1].groupby('cluster').size().sort_values(ascending=False)
    
    fig2 = go.Figure(go.Bar(
        x=[f"C{c}" for c in cluster_sizes.index],
        y=cluster_sizes.values,
        marker_color=cluster_sizes.values,
        marker_colorscale='Viridis',
    ))
    fig2.update_layout(
        title="Cluster Sizes",
        xaxis_title="Cluster ID",
        yaxis_title="Number of Sentences",
        width=1400,
        height=500,
    )
    
    bar_path = HTML_PATH.replace('.html', '_bars.html')
    fig2.write_html(bar_path)
    print(f"Saved bar chart: {bar_path}")


if __name__ == '__main__':
    create_visualization()
