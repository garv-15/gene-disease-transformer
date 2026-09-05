from pathlib import Path
import sys

import torch
from torch.optim import AdamW

# Allow running this file directly from the project root.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.gene_graph import load_gene_similarity
from src.models.gene_encoder import GeneEncoder
from src.models.gene_pretrainer import (
    GeneSimilarityDecoder,
    gene_similarity_loss,
)


DATA_PATH = "data/gene_similarity_matrix_cosine.txt"

TOP_K = 32
EMBEDDING_DIM = 128
HEADS = 4
NUM_LAYERS = 3

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Load biological gene graph
    # --------------------------------------------------

    genes, edge_index, edge_weight = load_gene_similarity(
        DATA_PATH,
        top_k=TOP_K,
    )

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    encoder = GeneEncoder(
        num_genes=len(genes),
        embedding_dim=EMBEDDING_DIM,
        heads=HEADS,
        num_layers=NUM_LAYERS,
        dropout=0.1,
    ).to(device)

    decoder = GeneSimilarityDecoder(
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    optimizer = AdamW(
        list(encoder.parameters())
        + list(decoder.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    encoder.train()
    decoder.train()

    for epoch in range(1, EPOCHS + 1):

        optimizer.zero_grad()

        gene_embeddings, _ = encoder(
            edge_index,
            edge_weight,
        )

        predicted_similarity = decoder(
            gene_embeddings,
            edge_index,
        )

        loss = gene_similarity_loss(
            predicted_similarity,
            edge_weight,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters())
            + list(decoder.parameters()),
            max_norm=1.0,
        )

        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} "
                f"| Loss: {loss.item():.6f}"
            )

    # --------------------------------------------------
    # Save trained encoder
    # --------------------------------------------------

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "gene_encoder.pt"

    torch.save(
        {
            "genes": genes,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "embedding_dim": EMBEDDING_DIM,
            "heads": HEADS,
            "num_layers": NUM_LAYERS,
            "top_k": TOP_K,
        },
        checkpoint_path,
    )

    print(f"\nSaved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()