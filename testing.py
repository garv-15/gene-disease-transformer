from data.gene_graph import load_gene_similarity
genes, edge_index, edge_weight = load_gene_similarity(
    "data/gene_similarity_matrix_cosine.txt",
    top_k=32,
)

print("Genes:", len(genes))
print("Edges:", edge_index.shape)
print("Weights:", edge_weight.shape)