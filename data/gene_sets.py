from pathlib import Path


def load_gene_sets(
    path: str | Path,
    gene_to_index: dict[str, int],
):
    """
    Load disease-centered gene sets from a GMT file.
    -------
    set_names:
        List of disease names
    set_doids:
        List of Disease Ontology IDs.
    gene_set_membership:
        List of (gene_index, set_index) tuples.
    """

    path = Path(path)

    set_names = []
    set_doids = []
    memberships = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 3:
                continue

            disease_name = parts[0].strip()
            doid = parts[1].strip()

            genes = [
                gene.strip()
                for gene in parts[2:]
                if gene.strip()
            ]

            set_index = len(set_names)

            set_names.append(disease_name)
            set_doids.append(doid)

            for gene in genes:
                if gene in gene_to_index:
                    memberships.append(
                        (
                            gene_to_index[gene],
                            set_index,
                        )
                    )

    return set_names, set_doids, memberships