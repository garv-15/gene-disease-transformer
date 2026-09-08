from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

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


DATA_PATH = "data/attribute_similarity_matrix_cosine.txt"
CHECKPOINT = "checkpoints/disease_encoder.pt"

POSITIVE_K = 16
NEGATIVE_K = 16

SEED = 42
TEST_RATIO = 0.20


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    (diseases, edge_index, edge_weight,) = load_disease_similarity_training_pairs(
        DATA_PATH,
        positive_k=POSITIVE_K,
        negative_k=NEGATIVE_K,
    )

    num_pairs = edge_weight.shape[0]
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(
        num_pairs,
        generator=generator,
    )

    test_size = int(
        num_pairs * TEST_RATIO
    )

    test_indices = permutation[:test_size]
    test_edge_index = edge_index[
        :, test_indices
    ].to(device)

    test_target = edge_weight[
        test_indices
    ].to(device)

    encoder = DiseaseEncoder(
        num_diseases=len(diseases),
        embedding_dim=checkpoint["embedding_dim"],
        heads=checkpoint["heads"],
        num_layers=checkpoint["num_layers"],
        dropout=0.0,
    ).to(device)

    decoder = DiseaseSimilarityDecoder(
        embedding_dim=checkpoint["embedding_dim"],
    ).to(device)

    encoder.load_state_dict(
        checkpoint["encoder_state_dict"]
    )

    decoder.load_state_dict(
        checkpoint["decoder_state_dict"]
    )

    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        full_edge_index = edge_index.to(device)
        full_edge_weight = edge_weight.to(device)

        embeddings, _ = encoder(
            full_edge_index,
            full_edge_weight,
        )

        predicted = decoder(
            embeddings,
            test_edge_index,
        )

    y_true = test_target.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    pearson = pearsonr(
        y_true,
        y_pred,
    ).statistic

    spearman = spearmanr(
        y_true,
        y_pred,
    ).statistic

    print(f"Device: {device}")
    print(f"Test pairs: {len(y_true)}")
    print(f"MAE: {mae:.6f}")
    print(f"Pearson: {pearson:.6f}")
    print(f"Spearman: {spearman:.6f}")


if __name__ == "__main__":
    main()