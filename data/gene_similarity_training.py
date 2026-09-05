from pathlib import Path

import numpy as np
import torch

EXPECTED_GENES = 4055

def load_gene_similarity_training_pairs(
    path: str | Path,
    positive_k: int = 32,
    negative_k: int = 32,
) -> tuple[
    list[str],
    torch.Tensor,
    torch.Tensor,
]:
    """
    build positive and low-similarity training pairs from the
    full gene-gene cosine similarity matrix.

    For every gene:
        positive_k  = highest-similarity genes
        negative_k  = lowest-similarity genes

    returns
    gene_symbols:
        gene symbols in matrix order.

    edge_index:
        shape [2, num_pairs].

    edge_weight:
        actual cosine similarity for each pair.
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

    if len(rows) < EXPECTED_GENES + 3:
        raise ValueError(
            f"Expected at least {EXPECTED_GENES + 3} rows, "
            f"found {len(rows)}."
        )

    gene_symbols = rows[0][3:]

    if len(gene_symbols) != EXPECTED_GENES:
        raise ValueError(
            f"Expected {EXPECTED_GENES} gene columns, "
            f"found {len(gene_symbols)}."
        )

    data_rows = rows[
        3:3 + EXPECTED_GENES
    ]

    if len(data_rows) != EXPECTED_GENES:
        raise ValueError(
            f"Expected {EXPECTED_GENES} gene rows, "
            f"found {len(data_rows)}."
        )

    row_gene_symbols = [
        row[0]
        for row in data_rows
    ]

    if row_gene_symbols != gene_symbols:
        raise ValueError(
            "Gene ordering differs between rows and columns."
        )

    matrix = np.empty(
        (EXPECTED_GENES, EXPECTED_GENES),
        dtype=np.float32,
    )

    for i, row in enumerate(data_rows):

        values = row[3:3 + EXPECTED_GENES]

        if len(values) != EXPECTED_GENES:
            raise ValueError(
                f"Row {i} contains {len(values)} values; "
                f"expected {EXPECTED_GENES}."
            )

        matrix[i] = np.asarray(
            values,
            dtype=np.float32,
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Gene similarity matrix contains NaN or infinite values."
        )

    np.fill_diagonal(matrix, np.nan)

    if positive_k + negative_k >= EXPECTED_GENES:
        raise ValueError(
            "positive_k + negative_k must be smaller "
            "than the number of genes."
        )


    positive_neighbors = np.argpartition(
        np.nan_to_num(
            matrix,
            nan=-np.inf,
        ),
        -positive_k,
        axis=1,
    )[:, -positive_k:]


    negative_neighbors = np.argpartition(
        np.nan_to_num(
            matrix,
            nan=np.inf,
        ),
        negative_k - 1,
        axis=1,
    )[:, :negative_k]

    source_positive = np.repeat(
        np.arange(EXPECTED_GENES),
        positive_k,
    )

    target_positive = positive_neighbors.reshape(-1)

    weight_positive = matrix[
        source_positive,
        target_positive,
    ]

    source_negative = np.repeat(
        np.arange(EXPECTED_GENES),
        negative_k,
    )

    target_negative = negative_neighbors.reshape(-1)

    weight_negative = matrix[
        source_negative,
        target_negative,
    ]


    source = np.concatenate(
        [
            source_positive,
            source_negative,
        ]
    )

    target = np.concatenate([
            target_positive,
            target_negative,
    ])

    weight = np.concatenate([
            weight_positive,
            weight_negative,
    ])

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

    print(f"Positive pairs: {len(source_positive)}")
    print(f"Low-similarity pairs: {len(source_negative)}")
    print(f"Total training pairs: {len(weight)}")

    return (gene_symbols, edge_index, edge_weight,)