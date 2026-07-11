import pytest
import pandas as pd
import plotly.graph_objects as go
from utils.network_graph import plot_similarity_network


def test_plot_similarity_network_returns_plotly_figure():
    # Setup simple square similarity matrix
    data = {
        "doc1": [1.0, 0.85, 0.20],
        "doc2": [0.85, 1.0, 0.10],
        "doc3": [0.20, 0.10, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3"])
    
    fig = plot_similarity_network(df, threshold=0.75)
    
    assert isinstance(fig, go.Figure)
    # Check that there are traces in the graph
    assert len(fig.data) == 2  # edge_hover_trace, node_trace
    
    # Check that layout has shapes representing the edges
    # doc1 and doc2 are connected (0.85 >= 0.75), so 1 line shape should exist
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0]["type"] == "line"


def test_plot_similarity_network_no_edges():
    # Setup matrix where no similarities exceed the threshold
    data = {
        "doc1": [1.0, 0.10, 0.20],
        "doc2": [0.10, 1.0, 0.15],
        "doc3": [0.20, 0.15, 1.0],
    }
    df = pd.DataFrame(data, index=["doc1", "doc2", "doc3"])
    
    fig = plot_similarity_network(df, threshold=0.75)
    
    assert isinstance(fig, go.Figure)
    # No shapes/lines should be added
    assert len(fig.layout.shapes) == 0
