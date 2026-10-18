# Node Classification on Citation Networks using Graph Neural Networks

This project compares four graph neural network (GNN) models for classifying research papers in citation networks:

- Graph Convolutional Network (GCN)
- GraphSAGE
- Graph Attention Network (GAT)
- A hybrid model that uses two GCN layers followed by a GAT classification layer

The work was carried out as a postgraduate dissertation project at IISER Tirupati. The main notebook runs in Google Colab and uses PyTorch Geometric.

## Why this project?

A research paper is not just a row of text features. It also cites other papers, and those links can provide useful context. This project tests whether combining paper features with the citation graph helps predict a paper's research area.

## Project files

```text
gnn-node-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── GNN_NodeClassification.ipynb
├── reports/
│   └── dissertation.pdf
└── docs/
    └── methodology.md
```

- `notebooks/GNN_NodeClassification.ipynb`: data exploration, model definitions, training, evaluation, sensitivity experiments, Citeseer evaluation, and embedding visualisation.
- `reports/dissertation.pdf`: the submitted project report.
- `docs/methodology.md`: a short guide to the models, datasets, reported results, and current limitations.

## Models

| Model | Main idea |
|---|---|
| GCN | Aggregates information from a node's neighbours using graph convolution. |
| GraphSAGE | Learns a neighbourhood aggregation function. |
| GAT | Learns attention weights to give neighbours different levels of importance. |
| Hybrid GCN+GAT | Uses two GCN layers for neighbourhood aggregation, followed by a GAT layer for classification. |

## Data

The notebook uses the Cora and Citeseer citation networks from the Planetoid datasets available through PyTorch Geometric. The notebook downloads the datasets when it runs; the raw datasets are not stored in this repository.

The dissertation reports these dataset details:

| Dataset | Nodes | Edges | Features | Classes |
|---|---:|---:|---:|---:|
| Cora | 2,708 | 10,556 | 1,433 | 7 |
| Citeseer | 3,327 | 9,104 | 3,703 | 6 |

Dataset edge-count conventions can differ between tools, so use the values printed by the notebook when reproducing the experiment.

## Run the notebook

### Recommended: Google Colab

1. Open [`GNN_NodeClassification.ipynb`](notebooks/GNN_NodeClassification.ipynb) in this repository.
2. Select **Open in Colab** if the GitHub/Colab interface offers it, or download the notebook and upload it to [Google Colab](https://colab.research.google.com/).
3. Select a runtime. A GPU can reduce training time, but a GPU is not essential for the small benchmark graphs.
4. Run the notebook cells from top to bottom.
5. Check the printed validation and test results, plots, and warnings.

The first setup cell installs PyTorch Geometric dependencies based on the active Colab PyTorch/CUDA environment. Cloud runtimes change over time, so an old package combination may need adjustment.

### Local environment

Python 3.10 was used in the dissertation. Install the packages listed in `requirements.txt`, then launch Jupyter and run the notebook. For PyTorch Geometric, follow the [official installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the default installation does not match your PyTorch and CUDA versions.

## Results reported in the dissertation

The dissertation reports the following test accuracies. These are the values documented in the report, not a guarantee that a fresh run will reproduce them exactly.

| Model | Cora test accuracy | Citeseer test accuracy |
|---|---:|---:|
| GCN | 81.50% | 71.00% |
| GraphSAGE | 81.00% | 70.30% |
| GAT | 82.60% | 70.00% |
| Hybrid GCN+GAT | 82.80% | 72.00% |

The report also gives a Davies–Bouldin index of 1.0495 for the hybrid model's t-SNE embeddings. See the dissertation for the full tables, figures, and discussion.

## Important notes before treating the results as final

This repository keeps the original notebook and dissertation intact. Before using the numbers as a final research result, I recommend checking the following:

1. **Test-set use during tuning:** the sensitivity-analysis helper in the notebook returns test accuracy for each setting. Hyperparameters should be selected using validation data, with the test set used only for the final evaluation.
2. **Class-name mapping:** check that the displayed Cora class names match the numeric labels returned by the installed Planetoid dataset. A mismatch can make plots and confusion-matrix labels misleading even when the model predictions are unchanged.
3. **Repeatability:** run each model with more than one random seed and report mean and standard deviation before making strong claims about small differences between models.
4. **Novelty claim:** the hybrid arrangement is the project's proposed design. A claim that it is novel should only be made after a careful literature review and comparison with closely related architectures.
5. **Runtime and package versions:** Colab's default software changes. Record the actual Python, PyTorch, PyTorch Geometric, and CUDA versions used for each final run.

These are research checks, not reasons to discard the project. They make the results easier for another person to trust and reproduce.

## Research questions

- How do GCN, GraphSAGE, and GAT compare on the Cora citation network?
- Does the hybrid GCN+GAT model improve on the individual baselines?
- Do the same model rankings hold on Citeseer?
- What do the learned node embeddings show about class separation?

## Citation

If you use this work, cite the dissertation and the original papers listed in its bibliography. The full report is included in `reports/dissertation.pdf`.

## License

No open-source license has been added yet. The dissertation itself includes an institute copyright notice. Do not assume that the report or all included material is freely reusable. Add a license only after confirming that you have the right to do so.
