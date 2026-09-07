## Project goal

Build a **biology-structured Transformer in PyTorch/PyTorch Geometric** that predicts gene–disease associations while providing biologically meaningful explanations.

The intended model should:

1. Predict whether a given **gene ↔ disease** association exists.
2. Use **gene–gene similarity** and **disease–disease similarity** to constrain information flow.
3. Use the disease/gene-set structure from the GMT files.
4. Preserve interpretable attention over biological relationships.
5. Provide stronger explanations using attention plus attribution/ablation methods rather than treating attention alone as proof of causality.

## Dataset currently understood

Core dimensions:

```text
Genes:                     4,055
Diseases:                    350
Gene–disease edges:       35,165 rows
Gene similarity:       4,055 × 4,055
Disease similarity:        350 × 350
Gene–disease matrix:    4,055 × 350
Disease gene sets:           350
Gene-set memberships:    35,164
```

Matrix semantics:

```text
1 / >0  = known association
0       = explicitly no association
NA      = unknown / not annotated
```

Important file-format details were resolved:

`gene_similarity_matrix_cosine.txt` contains 3 metadata rows, then the 4,055 × 4,055 numerical matrix.

`gene_attribute_matrix_*` contains gene metadata followed by disease-association columns.

`gene_set_library_crisp.gmt` is structured as:

```text
disease_name    DOID    GENE1    GENE2    GENE3 ...
```

`ENSP...` values in the metadata are **Ensembl protein IDs**, not Ensembl gene IDs.

`processing_scripts.tar` is legacy MATLAB/Octave material and is intentionally not being used.

## Completed

### 1. Gene similarity parser

Created:

`src/data/gene_graph.py`

It successfully:

* loads the 4,055 genes
* extracts the 4,055 × 4,055 numerical similarity matrix
* removes self-similarity
* constructs a sparse top-32 gene-neighbor graph

Current graph:

```text
Nodes: 4,055
Edges: 129,760
```

### 2. Gene Transformer encoder

Created:

`src/models/gene_encoder.py`

Current architecture:

```text
Gene ID
  ↓
Learnable embedding
  ↓
3 × TransformerConv layers
  ↓
128-dimensional gene representation
```

Configuration:

```text
embedding_dim = 128
attention heads = 4
layers = 3
dropout = 0.1
```

Each Transformer layer uses the gene similarity value as an edge feature.

The encoder also returns:

```text
edge_index
attention weights
```

for every layer.

Verified successfully:

```text
Gene embeddings: 4055 × 128

Layer 1 attention: 129760 × 4
Layer 2 attention: 129760 × 4
Layer 3 attention: 129760 × 4
```

### 3. Gene-set parser

Created:

`src/data/gene_sets.py`

Successfully loaded:

```text
350 disease gene sets
35,164 gene-set memberships
```

Examples include:

```text
acquired immunodeficiency syndrome → DOID:635
acquired metabolic disease          → DOID:0060158
adenocarcinoma                      → DOID:299
```

The GMT data is currently parsed but **not yet injected into the encoder**, deliberately avoiding premature leakage of gene–disease information.

### 4. Initial gene-encoder pretraining

Created:

`src/models/gene_pretrainer.py`

and:

`src/train/train_gene_encoder.py`

The encoder was trained to reconstruct observed gene–gene cosine similarity.

Training result:

```text
Epoch   1:  0.327373
Epoch  10:  0.025197
Epoch  50:  0.014741
Epoch 100:  0.005371
```

Checkpoint saved as:

`checkpoints/gene_encoder.pt`


### 5. Improve gene pretraining

Current pretraining uses only the **top-32 similar genes** for every gene.

This needs negative/low-similarity pairs so the embedding space learns:

```text
similar genes     → close
dissimilar genes  → separated
```
rather than potentially making too many genes similar.

> checkpoint to be updated

## Not completed yet


### 1. Disease encoder

Still to implement:

```text
350 disease nodes
      ↓
disease embeddings
      ↓
disease-disease similarity attention
      ↓
disease representations
```

using:

`attribute_similarity_matrix_cosine.txt`

### 2. Gene–disease prediction model

Still to implement the core task:

```text
Gene representation
        ↕
Cross-attention
        ↕
Disease representation
        ↓
P(gene, disease)
```

Output should ultimately represent the complete:

```text
4,055 × 350
```

gene–disease score matrix.

### 3. Correct handling of `NA`

Training/evaluation must use:

```text
1  → positive
0  → negative
NA → ignored
```

The loss must therefore be masked rather than treating `NA` as zero.

### 4. Data splitting / leakage prevention

This is especially important because the GMT sets and edge file contain the same biological associations.

We need controlled splits such as:

```text
random edge split
gene-held-out split
disease-held-out split
```

and potentially disease-family-held-out evaluation.

The GMT information must not accidentally reveal the target association during validation/test.

### 5. Biological module layer

The disease-centered gene sets need to be incorporated into the architecture in a way that does not leak target labels.

Potential structure:

```text
Genes
 ↓
Gene representation
 ↓
Gene-set/module representation
 ↓
Disease representation
```

### 6. Explanation system

Not implemented yet.

Final system should expose:

```text
prediction
  ↓
important gene relationships
  ↓
attention
  ↓
biological gene sets/modules
  ↓
disease relationship evidence
```

Attention should be supplemented with **Integrated Gradients / gradient-based attribution / ablation** so that explanations are not based solely on attention values.

## Current status

The project is currently at:

**Gene-side representation learning complete as a functional prototype.**

The validated pipeline is:

```text
gene similarity matrix
        ↓
top-32 biological gene graph
        ↓
3-layer Transformer
        ↓
4,055 × 128 gene embeddings
        ↓
attention weights retained
        ↓
gene similarity pretraining
```

// TODO: 

The major remaining work starts with **disease encoder and gene–disease cross-attention/link-prediction system**.

