# Methodology and experiment notes

## Goal

The project studies semi-supervised node classification on citation graphs. Each node represents a paper, its feature vector represents paper attributes, and edges represent citation links. The models predict one of the paper-area labels using node features and graph neighbourhood information.

## Models compared

- **GCN:** two graph convolution layers.
- **GraphSAGE:** two SAGE convolution layers using mean aggregation.
- **GAT:** graph attention layers with multi-head attention in the hidden layer.
- **Hybrid GCN+GAT:** two GCN layers followed by a one-head GAT classification layer.

## Datasets

The notebook loads Cora and Citeseer through `torch_geometric.datasets.Planetoid`, with feature normalization. It uses the dataset-provided train, validation, and test masks.

## Evaluation reported in the dissertation

| Model | Cora test accuracy | Citeseer test accuracy |
|---|---:|---:|
| GCN | 81.50% | 71.00% |
| GraphSAGE | 81.00% | 70.30% |
| GAT | 82.60% | 70.00% |
| Hybrid GCN+GAT | 82.80% | 72.00% |

The report also describes t-SNE visualisation of learned embeddings and uses the Davies–Bouldin index as a cluster-separation measure.

## Checks recommended before final publication

- Use validation performance to choose hyperparameters. Do not choose configurations based on test accuracy.
- Verify the numeric-label-to-class-name mapping before interpreting charts and confusion matrices.
- Repeat experiments across multiple random seeds and report variability.
- Record the exact software versions and runtime used.
- Compare the hybrid design with related work before describing it as novel.
