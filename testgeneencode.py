import torch

from data.gene_graph import load_gene_similarity
from src.models.gene_encoder import GeneEncoder


genes, edge_index, edge_weight = load_gene_similarity(
    "data/gene_similarity_matrix_cosine.txt",
    top_k=32,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

model = GeneEncoder(
    num_genes=len(genes),
    embedding_dim=128,
    heads=4,
    num_layers=3,
).to(device)

model.eval()

with torch.no_grad():
    gene_embeddings, attentions = model(
        edge_index,
        edge_weight,
    )

print("Device:", device)
print("Gene embeddings:", gene_embeddings.shape)
print("Number of attention layers:", len(attentions))

for i, attention in enumerate(attentions):
    edge_idx, weights = attention

    print(
        f"Layer {i + 1}:",
        "edges =", edge_idx.shape,
        "attention =", weights.shape,
    )