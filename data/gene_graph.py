from pathlib import Path

import numpy as np
import torch


EXPECTED_GENES = 4055


def load_gene_similarity(
    path: str | Path,
    top_k: int = 32,
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

    if len(rows) < EXPECTED_GENES + 3:
        raise ValueError(
            f"Expected at least {EXPECTED_GENES + 3} rows, "
            f"found {len(rows)}."
        )

    
    # Header
    

    # Row 0:
    # # | # | GeneSym | UNC13C | USP36 | AAK1 | ...
    gene_symbols = rows[0][3:]
    if len(gene_symbols) != EXPECTED_GENES:
        raise ValueError(
            f"Expected {EXPECTED_GENES} gene columns, "
            f"found {len(gene_symbols)}."
        )

    
    # Gene metadata
    

    # Row 1:
    # # | # | Ensemble Acc | ENSP... | ENSP... | ...
    # Row 2:
    # GeneSym | Ensemble Acc | GeneID/GeneID | 440279 | ...
    

    data_start = 3
    data_rows = rows[data_start:data_start + EXPECTED_GENES]
    if len(data_rows) != EXPECTED_GENES:
        raise ValueError(
            f"Expected {EXPECTED_GENES} data rows, "
            f"found {len(data_rows)}."
        )

    # ---------------------------------------------------------
    # Verify row gene ordering
    # ---------------------------------------------------------

    row_gene_symbols = [
        row[0]
        for row in data_rows
    ]

    if row_gene_symbols != gene_symbols:
        raise ValueError(
            "Gene ordering differs between rows and columns."
        )

    # ---------------------------------------------------------
    # Extract numerical similarity matrix
    # ---------------------------------------------------------

    matrix = np.empty(
        (EXPECTED_GENES, EXPECTED_GENES),
        dtype=np.float32,
    )

    for i, row in enumerate(data_rows):

        values = row[3:]

        if len(values) != EXPECTED_GENES:
            raise ValueError(
                f"Gene {row[0]} has {len(values)} "
                f"similarity values; expected {EXPECTED_GENES}."
            )

        try:
            matrix[i] = np.asarray(
                values,
                dtype=np.float32,
            )
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric similarity value found "
                f"for gene {row[0]!r}."
            ) from exc

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Similarity matrix contains NaN or infinite values."
        )

    print(
        f"Loaded gene similarity matrix: "
        f"{matrix.shape}"
    )

    # build sparse top-k graph

    # do not allow a gene to select itself.
    np.fill_diagonal(matrix, -np.inf)

    if not 1 <= top_k < EXPECTED_GENES:
        raise ValueError(
            f"top_k must be between "
            f"1 and {EXPECTED_GENES - 1}."
        )

    neighbors = np.argpartition(
        matrix,
        -top_k,
        axis=1,
    )[:, -top_k:]

    source = np.repeat(
        np.arange(EXPECTED_GENES),
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
        gene_symbols,
        edge_index,
        edge_weight,
    )