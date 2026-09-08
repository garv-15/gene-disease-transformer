import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseSimilarityDecoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        disease_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        z = self.projection(disease_embeddings)

        z = F.normalize(
            z,
            p=2,
            dim=-1,
        )

        source = edge_index[0]
        target = edge_index[1]

        similarity = (
            z[source] * z[target]
        ).sum(dim=-1)

        return similarity


def disease_similarity_loss(
    predicted_similarity: torch.Tensor,
    target_similarity: torch.Tensor,
) -> torch.Tensor:

    return F.mse_loss(
        predicted_similarity,
        target_similarity,
    )