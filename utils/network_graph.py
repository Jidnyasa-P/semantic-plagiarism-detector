"""
network_graph.py
----------------
Generates interactive document plagiarism network graphs using networkx and Plotly.
Documents are represented as nodes, and similarities above the threshold are edges.
"""

import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_similarity_network(
    similarity_df: pd.DataFrame,
    threshold: float = 0.59,
    title: str = "Document Plagiarism Network"
) -> go.Figure:
    """
    Builds a networkx graph from the similarity matrix and returns an interactive Plotly figure.
    
    Args:
        similarity_df: Square N×N DataFrame of similarity scores.
        threshold:     Edge threshold; pairs with similarity >= threshold are connected.
        title:         Title of the graph.
        
    Returns:
        Plotly Graph Objects Figure.
    """
    # Create networkx graph
    G = nx.Graph()
    
    # Add all documents as nodes
    doc_names = list(similarity_df.columns)
    for name in doc_names:
        G.add_node(name)
        
    # Add edges for pairs exceeding threshold
    n = len(doc_names)
    edge_similarities = {}
    for i in range(n):
        for j in range(i + 1, n):
            score = float(similarity_df.iloc[i, j])
            if score >= threshold:
                G.add_edge(doc_names[i], doc_names[j])
                edge_similarities[(doc_names[i], doc_names[j])] = score

    # Compute layout coordinates (spring layout forces connected nodes closer)
    # Seed layout for reproducibility
    pos = nx.spring_layout(G, seed=42, k=1.0 / np.sqrt(max(1, len(G.nodes()))))
    
    # ── Draw Edges (using Plotly shapes for custom colors/widths) ─────────────────
    shapes = []
    # For hover info, we can also add a transparent trace under each edge
    edge_trace_x = []
    edge_trace_y = []
    edge_hover_texts = []
    
    for edge in G.edges():
        doc_a, doc_b = edge
        x0, y0 = pos[doc_a]
        x1, y1 = pos[doc_b]
        
        # Get similarity score
        score = edge_similarities.get((doc_a, doc_b), edge_similarities.get((doc_b, doc_a), threshold))
        
        # Line width based on similarity
        line_width = max(1.5, score * 6.0)
        
        # Color based on severity
        if score >= 0.90:
            color = "#ff4b4b"  # High (Red)
        elif score >= 0.75:
            color = "#ffa500"  # Medium (Orange)
        else:
            color = "#ffd700"  # Low-moderate above threshold (Yellow)
            
        shapes.append(dict(
            type="line",
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            line=dict(color=color, width=line_width),
            layer="below"
        ))
        
        # Add to hover trace (midpoint of the edge for tooltip)
        edge_trace_x.extend([x0, x1, None])
        edge_trace_y.extend([y0, y1, None])
        edge_hover_texts.append(f"<b>Match:</b> {doc_a} ↔ {doc_b}<br><b>Similarity:</b> {score:.1%}")

    # Hidden scatter trace to enable hover text on edges (hovering on midpoints)
    edge_hover_x = []
    edge_hover_y = []
    for edge in G.edges():
        doc_a, doc_b = edge
        x0, y0 = pos[doc_a]
        x1, y1 = pos[doc_b]
        # Midpoint coordinate
        edge_hover_x.append((x0 + x1) / 2.0)
        edge_hover_y.append((y0 + y1) / 2.0)
        
    edge_hover_trace = go.Scatter(
        x=edge_hover_x, y=edge_hover_y,
        mode="markers",
        marker=dict(size=8, color="rgba(0,0,0,0)"),  # Invisible markers
        text=edge_hover_texts,
        hoverinfo="text",
        name="Connections"
    )

    # ── Draw Nodes ────────────────────────────────────────────────────────────────
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        deg = G.degree(node)
        
        # Size based on degree (number of suspicious connections)
        node_size.append(20 + deg * 6)
        
        # Color based on degree
        if deg == 0:
            node_color.append("#2e7d32")  # Clean (Green)
        elif deg == 1:
            node_color.append("#f9a825")  # Warning (Yellow-orange)
        else:
            node_color.append("#c62828")  # Plagiarism cluster (Red)
            
        node_hover.append(
            f"<b>📄 Document:</b> {node}<br>"
            f"<b>🚨 Flagged connections:</b> {deg} / {len(doc_names)-1}"
        )
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[name.split(".")[0] for name in node_text],  # Short display name (no extension)
        textposition="top center",
        hoverinfo="text",
        textfont=dict(color="#e6edf3", size=10, family="Arial Black"),
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=2, color="#ffffff")
        ),
        hovertext=node_hover,
        name="Documents"
    )
    
    # ── Figure Layout ─────────────────────────────────────────────────────────────
    fig = go.Figure(
        data=[edge_hover_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16, family="Arial Black")),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=40, l=40, r=40, t=50),
            shapes=shapes,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#e6edf3")
        )
    )
    
    return fig
