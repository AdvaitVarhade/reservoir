"""Watershed-to-Reservoir graph encoder.

The graph can represent upstream watersheds, tributaries, inlet regions,
erosion hotspots, and reservoir zones. If a detailed river network is not
available, the module builds a downstream chain graph from node order.
"""

import math

import torch
from torch import nn


class GraphAttentionBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.norm = nn.LayerNorm(hidden)

    def forward(self, states: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(states)
        key = self.key(states)
        value = self.value(states)
        scale = math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-1, -2)) / max(scale, 1.0)
        scores = scores.masked_fill(adjacency <= 0, -1e4)
        attention = torch.softmax(scores, dim=-1)
        updated = torch.matmul(attention, value)
        updated = self.norm(states + self.output(updated))
        return updated, attention


class WatershedReservoirGNN(nn.Module):
    """Directed message passing from upstream nodes toward reservoir zones."""

    def __init__(self, features: int, hidden: int, layers: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(features, hidden)
        self.layers = nn.ModuleList(GraphAttentionBlock(hidden) for _ in range(layers))
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden))

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if adjacency is None:
            adjacency = self._fallback_adjacency(nodes)
        adjacency = adjacency.float()
        adjacency = torch.clamp(adjacency + torch.eye(adjacency.size(-1), device=adjacency.device).unsqueeze(0), min=0.0, max=1.0)
        states = torch.relu(self.input_projection(nodes))
        attentions = []
        for layer in self.layers:
            states, attention = layer(states, adjacency)
            attentions.append(attention)
        pooled = self.readout(states.mean(dim=1))
        return pooled, {"node_embeddings": states, "graph_attention": torch.stack(attentions, dim=1)}

    @staticmethod
    def _fallback_adjacency(nodes: torch.Tensor) -> torch.Tensor:
        batch, node_count, _ = nodes.shape
        adjacency = torch.zeros(batch, node_count, node_count, device=nodes.device)
        for idx in range(node_count):
            adjacency[:, idx, idx] = 1.0
            if idx + 1 < node_count:
                adjacency[:, idx, idx + 1] = 1.0
                adjacency[:, idx + 1, idx] = 0.3
        return adjacency
