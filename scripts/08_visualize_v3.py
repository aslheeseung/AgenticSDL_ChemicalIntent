"""
Phase 6 v3: Interactive Visualization for v3a results
"""
import os, numpy as np, pandas as pd, json
import plotly.express as px
import plotly.graph_objects as go

BASE = "/home/hs/oer-catalyst-project/output"

df = pd.read_csv(f"{BASE}/sentences.csv")
umap_2d = np.load(f"{BASE}/umap_2d_v3a.npy")
labels = np.load(f"{BASE}/clustering_labels_v3a.npy")

with open(f"{BASE}/clusters_summary_v3a.json") as f:
    summaries = json.load(f)

cluster_keywords = {}
for s in summaries:
    kws = s.get('top_keywords', [])[:5]
    cluster_keywords[s['cluster_id']] = ', '.join(kws)

df['x'] = umap_2d[:, 0]
df['y'] = umap_2d[:, 1]
df['cluster'] = labels
df['cluster_label'] = df['cluster'].apply(
    lambda c: f"C{c} ({cluster_keywords.get(c, 'noise')})" if c != -1 else "Noise"
)
df['hover'] = df.apply(
    lambda r: f"[{r['cluster_label']}]<br>Paper: {r['paper_id']}<br>Section: {r['section']}<br><b>{r['sentence'][:150]}</b>",
    axis=1
)

# 노이즈 재배정 (이미 된 상태지만 확인)
if (df['cluster'] == -1).sum() > 0:
    print(f"Warning: {(df['cluster']==-1).sum()} noise points remaining")

fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=df['x'], y=df['y'],
    mode='markers',
    marker=dict(
        size=3, color=df['cluster'],
        colorscale=px.colors.qualitative.Alphabet + px.colors.qualitative.Set3,
        opacity=0.6,
        colorbar=dict(title="Cluster", tickfont=dict(size=9)),
    ),
    text=df['hover'], hoverinfo='text',
))

fig.update_layout(
    title="OER Catalyst Synthesis - 37 Clusters (UMAP + HDBSCAN v3a)",
    xaxis_title="UMAP 1", yaxis_title="UMAP 2",
    width=1400, height=900, hovermode='closest',
)
fig.write_html(f"{BASE}/interactive_plot_v3a.html")
print(f"Saved: {BASE}/interactive_plot_v3a.html ({os.path.getsize(f'{BASE}/interactive_plot_v3a.html')/1024:.0f} KB)")

# 바 차트
sizes = df.groupby('cluster').size().sort_values(ascending=False)
fig2 = go.Figure(go.Bar(
    x=[f"C{c} ({cluster_keywords.get(c,'')[:30]})" for c in sizes.index],
    y=sizes.values, marker_color=sizes.values, marker_colorscale='Viridis',
))
fig2.update_layout(
    title="Cluster Sizes (v3a)", xaxis_title="Cluster", yaxis_title="Sentences",
    width=1600, height=500, xaxis_tickangle=-45,
)
fig2.write_html(f"{BASE}/interactive_plot_v3a_bars.html")
print(f"Saved: {BASE}/interactive_plot_v3a_bars.html")
