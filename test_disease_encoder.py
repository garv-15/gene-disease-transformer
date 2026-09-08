import torch

from data.disease_graph import load_disease_similarity
from src.models.disease_encoder import DiseaseEncoder

diseases, edge_index, edge_weight = load_disease_similarity(
    "data/attribute_similarity_matrix_cosine.txt",
    top_k=16,
)

device = torch.device(
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else "cpu"
)

edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

model = DiseaseEncoder(
    num_diseases=len(diseases),
    embedding_dim=128,
    heads=4,
    num_layers=3,
).to(device)

model.eval()

with torch.no_grad():
    disease_embeddings, attentions = model(
        edge_index,
        edge_weight,
    )

print("Device:", device)
print("Disease embeddings:", disease_embeddings.shape)
print("Number of attention layers:", len(attentions))

for i, attention in enumerate(attentions):
    edge_idx, weights = attention

    print(
        f"Layer {i + 1}:",
        "edges =", edge_idx.shape,
        "attention =", weights.shape,
    )