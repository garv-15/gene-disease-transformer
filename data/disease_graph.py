from pathlib import Path

import numpy as np
import torch


EXPECTED_DISEASES = 350


def load_disease_similarity(
    path: str | Path,
    top_k: int = 16,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:

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
    # Actual numerical matrix
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

        values = row[3:3 + EXPECTED_DISEASES]

        if len(values) != EXPECTED_DISEASES:
            raise ValueError(
                f"Disease {row[0]} has {len(values)} "
                f"similarity values; expected {EXPECTED_DISEASES}."
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
    # Sparse top-k disease graph
    # ---------------------------------------------------------

    np.fill_diagonal(matrix, -np.inf)

    if not 1 <= top_k < EXPECTED_DISEASES:
        raise ValueError(
            f"top_k must be between "
            f"1 and {EXPECTED_DISEASES - 1}."
        )

    neighbors = np.argpartition(
        matrix,
        -top_k,
        axis=1,
    )[:, -top_k:]

    source = np.repeat(
        np.arange(EXPECTED_DISEASES),
        top_k,
    )

    target = neighbors.reshape(-1)

    weight = matrix[
        source,
        target,
    ]

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

    return (
        disease_names,
        edge_index,
        edge_weight,
    )