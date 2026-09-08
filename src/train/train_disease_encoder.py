from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

from data.disease_similarity_training import (
    load_disease_similarity_training_pairs,
)

from src.models.disease_encoder import DiseaseEncoder
from src.models.disease_pretrainer import (
    DiseaseSimilarityDecoder,
)

DATA_PATH = ("data/attribute_similarity_matrix_cosine.txt")

POSITIVE_K = 16
NEGATIVE_K = 16

EMBEDDING_DIM = 128
HEADS = 4
NUM_LAYERS = 3

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100


def main():
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")
    print(f"Using device: {device}")
    
    (diseases, edge_index, edge_weight,) = load_disease_similarity_training_pairs(
        DATA_PATH,
        positive_k=POSITIVE_K,
        negative_k=NEGATIVE_K,
    )

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    encoder = DiseaseEncoder(
        num_diseases=len(diseases),
        embedding_dim=EMBEDDING_DIM,
        heads=HEADS,
        num_layers=NUM_LAYERS,
        dropout=0.1,
    ).to(device)

    decoder = DiseaseSimilarityDecoder(
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    optimizer = AdamW(
        list(encoder.parameters())
        + list(decoder.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    encoder.train()
    decoder.train()

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        disease_embeddings, _ = encoder(
            edge_index,
            edge_weight,
        )
        
        predicted_similarity = decoder(
            disease_embeddings,
            edge_index,
        )

        loss = F.mse_loss(
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
        encoder.eval()
        with torch.no_grad():
            disease_embeddings, attentions = encoder(
                edge_index,
                edge_weight,
            )
        torch.save(
            {
                "diseases": diseases,
                "embeddings": disease_embeddings.cpu(),
                "embedding_dim": EMBEDDING_DIM,
            },
            checkpoint_dir / "disease_embeddings.pt",
        )

    checkpoint_path = (
        checkpoint_dir
        / "disease_encoder.pt"
    )

    torch.save({
            "diseases": diseases,
            "encoder_state_dict":
                encoder.state_dict(),
            "decoder_state_dict":
                decoder.state_dict(),
            "embedding_dim":
                EMBEDDING_DIM,
            "heads":
                HEADS,
            "num_layers":
                NUM_LAYERS,
            "positive_k":
                POSITIVE_K,
            "negative_k":
                NEGATIVE_K,
        },
        checkpoint_path,
    )

    print(f"\nSaved checkpoint: "f"{checkpoint_path}")

if __name__ == "__main__":
    main()