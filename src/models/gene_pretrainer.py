import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneSimilarityDecoder(nn.Module):
    """
    Reconstructs known gene-gene similarity from learned
    gene representations.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        gene_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        z = self.projection(gene_embeddings)

        z = F.normalize(z, p=2, dim=-1)

        source = edge_index[0]
        target = edge_index[1]

        similarity = (
            z[source] * z[target]
        ).sum(dim=-1)

        return similarity


def gene_similarity_loss(
    predicted_similarity: torch.Tensor,
    target_similarity: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared error between predicted and observed
    gene-gene similarity.
    """

    return F.mse_loss(
        predicted_similarity,
        target_similarity,
    )