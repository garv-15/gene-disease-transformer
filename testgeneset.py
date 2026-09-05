from data.gene_graph import load_gene_similarity
from data.gene_sets import load_gene_sets


genes, _, _ = load_gene_similarity(
    "data/gene_similarity_matrix_cosine.txt",
)

gene_to_index = {
    gene: index
    for index, gene in enumerate(genes)
}

set_names, set_doids, memberships = load_gene_sets(
    "data/gene_set_library_crisp.gmt",
    gene_to_index,
)

print("Genes:", len(genes))
print("Gene sets:", len(set_names))
print("Memberships:", len(memberships))

for i in range(min(3, len(set_names))):
    print(
        set_names[i],
        set_doids[i],
    )