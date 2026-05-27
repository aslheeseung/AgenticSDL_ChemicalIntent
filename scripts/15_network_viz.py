"""
Phase 9b: Synthesis Network Interactive Visualization
Sankey diagram + Chord diagram + HTML 네트워크
"""
import os, json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

BASE = "/home/hs/oer-catalyst-project/output"

def create_sankey():
    """Sankey diagram: 공정 흐름"""
    with open(f"{BASE}/synthesis_network.json") as f:
        network = json.load(f)
    
    nodes = network['nodes']
    edges = network['edges']
    
    # 노드 인덱스 매핑
    node_ids = [n['id'] for n in nodes]
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    
    # 노드 크기 정규화
    max_count = max(n['count'] for n in nodes)
    
    # 색상 팔레트
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1
    
    # Sankey용 데이터
    sources, targets, values, labels = [], [], [], []
    link_colors = []
    
    for edge in edges:
        if edge['source'] in node_map and edge['target'] in node_map:
            sources.append(node_map[edge['source']])
            targets.append(node_map[edge['target']])
            values.append(edge['count'])
            # 투명도를 확률에 비례
            alpha = min(0.8, edge['probability'] + 0.2)
            link_colors.append(f'rgba(100,100,100,{alpha:.2f})')
    
    node_colors = [colors[i % len(colors)] for i in range(len(nodes))]
    
    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            pad=15,
            thickness=25,
            line=dict(color='black', width=0.5),
            label=[f"{n['id']}\n({n['count']})" for n in nodes],
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    ))
    
    fig.update_layout(
        title=dict(
            text="OER Catalyst Synthesis Route Network (905 Papers, Sankey Diagram)",
            font=dict(size=18),
        ),
        width=1600,
        height=900,
        font=dict(size=11),
    )
    
    fig.write_html(f"{BASE}/network_sankey.html")
    print(f"Saved: {BASE}/network_sankey.html")


def create_force_graph():
    """Force-directed network graph (인터랙티브)"""
    with open(f"{BASE}/synthesis_network.json") as f:
        network = json.load(f)
    
    nodes = network['nodes']
    edges = network['edges']
    
    # HTML 파일로 직접 생성 (D3-style with pure JS)
    max_count = max(n['count'] for n in nodes)
    
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OER Catalyst Synthesis Network</title>
<style>
body { margin: 0; background: #1a1a2e; font-family: 'Segoe UI', sans-serif; }
#tooltip {
    position: absolute; display: none; background: rgba(0,0,0,0.85);
    color: #fff; padding: 12px 16px; border-radius: 8px; font-size: 13px;
    max-width: 350px; pointer-events: none; z-index: 100;
    border: 1px solid rgba(255,255,255,0.2);
}
#legend {
    position: absolute; top: 20px; right: 20px; background: rgba(0,0,0,0.7);
    padding: 15px; border-radius: 10px; color: #fff; font-size: 12px;
}
#title {
    position: absolute; top: 15px; left: 20px; color: #fff; font-size: 20px;
    font-weight: bold;
}
#stats {
    position: absolute; bottom: 20px; left: 20px; background: rgba(0,0,0,0.7);
    padding: 12px; border-radius: 8px; color: #aaa; font-size: 11px;
}
svg { width: 100vw; height: 100vh; }
</style>
</head>
<body>
<div id="title">OER Catalyst Synthesis Route Network</div>
<div id="tooltip"></div>
<div id="legend"></div>
<div id="stats"></div>
<svg id="graph"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const nodes = """ + json.dumps(nodes) + """;
const edges = """ + json.dumps(edges) + """;

const width = window.innerWidth;
const height = window.innerHeight;

const color = d3.scaleOrdinal(d3.schemeTableau10);

const maxNodeCount = d3.max(nodes, d => d.count);
const maxEdgeCount = d3.max(edges, d => d.count);

const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(edges).id(d => d.id)
        .distance(d => 120 - d.count * 0.5)
        .strength(d => 0.3 + d.probability * 0.7))
    .force('charge', d3.forceManyBody().strength(-400))
    .force('center', d3.forceCenter(width/2, height/2))
    .force('collision', d3.forceCollide().radius(d => 15 + d.count/maxNodeCount * 30));

const svg = d3.select('#graph');

// Arrow marker
svg.append('defs').append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 20).attr('refY', 0)
    .attr('markerWidth', 6).attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#666');

const link = svg.append('g').selectAll('line')
    .data(edges).join('line')
    .attr('stroke', d => `rgba(150,150,200,${0.2 + d.probability * 0.6})`)
    .attr('stroke-width', d => 1 + d.count/maxEdgeCount * 6)
    .attr('marker-end', 'url(#arrow)');

const node = svg.append('g').selectAll('g')
    .data(nodes).join('g')
    .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

node.append('circle')
    .attr('r', d => 8 + d.count/maxNodeCount * 25)
    .attr('fill', (d, i) => color(i))
    .attr('stroke', '#fff').attr('stroke-width', 1.5)
    .attr('opacity', 0.85);

node.append('text')
    .text(d => d.id.replace(/_/g, ' '))
    .attr('dy', d => 15 + d.count/maxNodeCount * 25)
    .attr('text-anchor', 'middle')
    .attr('fill', '#ccc').attr('font-size', '10px')
    .attr('font-weight', 'bold');

// Tooltip
const tooltip = document.getElementById('tooltip');
node.on('mouseover', function(event, d) {
    const incoming = edges.filter(e => e.target.id === d.id);
    const outgoing = edges.filter(e => e.source.id === d.id);
    let html = `<b>${d.id.replace(/_/g,' ')}</b><br>Total: ${d.count} occurrences<br>`;
    if (outgoing.length > 0) {
        html += '<br><b>→ Next steps:</b>';
        outgoing.sort((a,b) => b.count - a.count).forEach(e => {
            html += `<br>  ${e.target.id.replace(/_/g,' ')}: ${e.count} (${(e.probability*100).toFixed(0)}%)`;
        });
    }
    if (incoming.length > 0) {
        html += '<br><br><b>← Previous steps:</b>';
        incoming.sort((a,b) => b.count - a.count).forEach(e => {
            html += `<br>  ${e.source.id.replace(/_/g,' ')}: ${e.count}`;
        });
    }
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    tooltip.style.left = (event.pageX + 15) + 'px';
    tooltip.style.top = (event.pageY - 10) + 'px';
}).on('mouseout', () => { tooltip.style.display = 'none'; });

simulation.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
});

function dragstarted(event) { if (!event.active) simulation.alphaTarget(0.3).restart(); event.subject.fx = event.subject.x; event.subject.fy = event.subject.y; }
function dragged(event) { event.subject.fx = event.x; event.subject.fy = event.y; }
function dragended(event) { if (!event.active) simulation.alphaTarget(0); event.subject.fx = null; event.subject.fy = null; }

// Legend
const legend = document.getElementById('legend');
legend.innerHTML = '<b>Node size</b>: occurrence count<br><b>Edge width</b>: transition count<br><b>Drag</b> nodes to rearrange<br><b>Hover</b> for details';

// Stats
const stats = document.getElementById('stats');
const topRoute = edges.sort((a,b) => b.count - a.count)[0];
stats.innerHTML = `Nodes: ${nodes.length} | Edges: ${edges.length}<br>Top transition: ${topRoute.source.id} → ${topRoute.target.id} (${topRoute.count}x)`;
</script>
</body>
</html>"""
    
    with open(f"{BASE}/network_interactive.html", 'w') as f:
        f.write(html)
    
    print(f"Saved: {BASE}/network_interactive.html")


def create_heatmap():
    """전이 확률 히트맵"""
    trans_df = pd.read_csv(f"{BASE}/transition_matrix.csv", index_col=0)
    
    # 확률로 정규화
    prob_df = trans_df.div(trans_df.sum(axis=1), axis=0).fillna(0)
    
    # 빈도 낮은 intent 제거
    intent_order = trans_df.sum(axis=1).sort_values(ascending=False).head(20).index
    prob_subset = prob_df.loc[intent_order, intent_order]
    
    fig = go.Figure(go.Heatmap(
        z=prob_subset.values,
        x=[c.replace('_', ' ') for c in prob_subset.columns],
        y=[c.replace('_', ' ') for c in prob_subset.index],
        colorscale='Blues',
        text=[[f'{v:.1%}' if v > 0.01 else '' for v in row] for row in prob_subset.values],
        texttemplate='%{text}',
        textfont={'size': 9},
        hovertemplate='%{y} → %{x}: %{z:.1%}<extra></extra>',
    ))
    
    fig.update_layout(
        title="Transition Probability Matrix (Top 20 Intents)",
        xaxis_title="Next Step →",
        yaxis_title="Current Step",
        width=900, height=900,
        xaxis_tickangle=-45,
    )
    
    fig.write_html(f"{BASE}/network_heatmap.html")
    print(f"Saved: {BASE}/network_heatmap.html")


if __name__ == '__main__':
    create_sankey()
    create_force_graph()
    create_heatmap()
    print("\nDone!")
