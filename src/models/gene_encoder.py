import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GeneEncoderLayer(nn.Module):
    """
    One biology-aware Transformer layer for gene representations.

    The graph structure comes from gene-gene similarity.
    Cosine similarity is supplied as an edge feature.
    """

    def __init__(
        self,
        embedding_dim: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()

        self.attention = TransformerConv(
            in_channels=embedding_dim,
            out_channels=embedding_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,
            beta=True,
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        residual = x

        edge_attr = edge_weight.unsqueeze(-1)

        x, attention_data = self.attention(
            x,
            edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )

        attention_edge_index, attention_weights = attention_data

        x = self.dropout(x)

        x = self.norm(
            x + residual
        )

        return (
            x,
            attention_edge_index,
            attention_weights,
        )


class GeneEncoder(nn.Module):
    """
    Biology-structured Transformer encoder for genes.

    Nodes:
        Genes

    Edges:
        Gene-gene similarity

    Edge features:
        Cosine similarity

    Output:
        Learned representation for every gene.
    """

    def __init__(
        self,
        num_genes: int,
        embedding_dim: int = 128,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        if embedding_dim % heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by heads."
            )

        self.embedding = nn.Embedding(
            num_embeddings=num_genes,
            embedding_dim=embedding_dim,
        )

        self.layers = nn.ModuleList(
            [
                GeneEncoderLayer(
                    embedding_dim=embedding_dim,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:

        num_genes = self.embedding.num_embeddings

        gene_ids = torch.arange(
            num_genes,
            device=edge_index.device,
        )

        x = self.embedding(gene_ids)

        attentions = []

        for layer in self.layers:

            (
                x,
                attention_edge_index,
                attention_weights,
            ) = layer(
                x,
                edge_index,
                edge_weight,
            )

            attentions.append(
                (
                    attention_edge_index,
                    attention_weights,
                )
            )

        return x, attentions