from pathlib import Path

import numpy as np
import torch


EXPECTED_DISEASES = 350


def load_disease_similarity_training_pairs(
    path: str | Path,
    positive_k: int = 16,
    negative_k: int = 16,
) -> tuple[
    list[str],
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build disease-disease training pairs from the full
    cosine similarity matrix.

    For each disease:
        positive_k -> highest-similarity diseases
        negative_k -> lowest-similarity diseases

    Returns
    -------
    disease_names:
        Disease names in matrix order.

    edge_index:
        Shape [2, num_pairs].

    edge_weight:
        Observed cosine similarity for each pair.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        rows = [
            line.rstrip("\n\r").split("\t")
            for line in f
            if line.strip()
        ]

    if len(rows) < EXPECTED_DISEASES + 3:
        raise ValueError(
            f"Expected at least {EXPECTED_DISEASES + 3} rows, "
            f"found {len(rows)}."
        )

    # ---------------------------------------------------------
    # Disease names
    # ---------------------------------------------------------

    disease_names = rows[0][3:]

    if len(disease_names) != EXPECTED_DISEASES:
        raise ValueError(
            f"Expected {EXPECTED_DISEASES} diseases in header, "
            f"found {len(disease_names)}."
        )

    # ---------------------------------------------------------
    # Numerical matrix
    # ---------------------------------------------------------

    data_rows = rows[
        3:3 + EXPECTED_DISEASES
    ]

    if len(data_rows) != EXPECTED_DISEASES:
        raise ValueError(
            f"Expected {EXPECTED_DISEASES} disease rows, "
            f"found {len(data_rows)}."
        )

    row_disease_names = [
        row[0]
        for row in data_rows
    ]

    if row_disease_names != disease_names:
        raise ValueError(
            "Disease ordering differs between rows and columns."
        )

    matrix = np.empty(
        (EXPECTED_DISEASES, EXPECTED_DISEASES),
        dtype=np.float32,
    )

    for i, row in enumerate(data_rows):

        values = row[
            3:3 + EXPECTED_DISEASES
        ]

        if len(values) != EXPECTED_DISEASES:
            raise ValueError(
                f"Row {i} contains {len(values)} values; "
                f"expected {EXPECTED_DISEASES}."
            )

        try:
            matrix[i] = np.asarray(
                values,
                dtype=np.float32,
            )
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric similarity value found "
                f"for disease {row[0]!r}."
            ) from exc

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Disease similarity matrix contains NaN "
            "or infinite values."
        )

    print(
        f"Loaded disease similarity matrix: {matrix.shape}"
    )

    # ---------------------------------------------------------
    # Remove self-similarity
    # ---------------------------------------------------------

    np.fill_diagonal(matrix, np.nan)

    if positive_k + negative_k >= EXPECTED_DISEASES:
        raise ValueError(
            "positive_k + negative_k must be smaller "
            "than the number of diseases."
        )

    # ---------------------------------------------------------
    # Highest-similarity pairs
    # ---------------------------------------------------------

    positive_neighbors = np.argpartition(
        np.nan_to_num(
            matrix,
            nan=-np.inf,
        ),
        -positive_k,
        axis=1,
    )[:, -positive_k:]

    # ---------------------------------------------------------
    # Lowest-similarity pairs
    # ---------------------------------------------------------

    negative_neighbors = np.argpartition(
        np.nan_to_num(
            matrix,
            nan=np.inf,
        ),
        negative_k - 1,
        axis=1,
    )[:, :negative_k]

    source_positive = np.repeat(
        np.arange(EXPECTED_DISEASES),
        positive_k,
    )

    target_positive = (
        positive_neighbors.reshape(-1)
    )

    weight_positive = matrix[
        source_positive,
        target_positive,
    ]

    source_negative = np.repeat(
        np.arange(EXPECTED_DISEASES),
        negative_k,
    )

    target_negative = (
        negative_neighbors.reshape(-1)
    )

    weight_negative = matrix[
        source_negative,
        target_negative,
    ]

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    source = np.concatenate(
        [
            source_positive,
            source_negative,
        ]
    )

    target = np.concatenate(
        [
            target_positive,
            target_negative,
        ]
    )

    weight = np.concatenate(
        [
            weight_positive,
            weight_negative,
        ]
    )

    valid = np.isfinite(weight)

    source = source[valid]
    target = target[valid]
    weight = weight[valid]

    edge_index = torch.tensor(
        np.stack([source, target]),
        dtype=torch.long,
    )

    edge_weight = torch.tensor(
        weight,
        dtype=torch.float32,
    )

    print(
        f"Positive pairs: "
        f"{len(source_positive)}"
    )

    print(
        f"Low-similarity pairs: "
        f"{len(source_negative)}"
    )

    print(
        f"Total training pairs: "
        f"{len(weight)}"
    )

    return (
        disease_names,
        edge_index,
        edge_weight,
    )