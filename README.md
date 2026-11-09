# Graph analytics and graph ML on two real networks

Team project for the course *Massive Graph Management & Analytics* at CentraleSupélec (November 2025 – February 2026). It has two parts:

1. **Network analysis.** Summary statistics, centrality, community detection and influence maximisation on an email network and a Twitch network.
2. **Graph machine learning.** Shallow node embeddings (spectral, Node2Vec) and graph neural networks (GAT, GraphSAGE) for node classification and link prediction on the same two graphs.

Forked from [Carda01/community_detection](https://github.com/Carda01/community_detection).

## Datasets

| Graph | Nodes | Edges | What a node / edge is | Labels and attributes |
|---|---|---|---|---|
| [email-Eu-core](https://snap.stanford.edu/data/email-Eu-core.html) (SNAP) | 986 | 16,064 | Members of a European research institution; an edge means at least one email between two members | Department of each member (42 departments), used as ground truth |
| [Twitch PTBR](https://snap.stanford.edu/data/twitch-social-networks.html) (SNAP, Rozemberczki et al., 2019) | 1,912 | 31,299 | Portuguese-speaking Twitch users; an edge is a mutual follow | Account age, total views, `mature`, `partner` |

The email graph is loaded as undirected, with self-loops and the nodes left isolated after their removal dropped, which is why it has 986 nodes rather than SNAP's 1,005. Both graphs form a single connected component.

## Repository layout

```
.
├── utils.py                                   # shared functions for the part 1 notebooks
├── requirements.txt
├── analyses/                                  # part 1: network analysis
│   ├── experiments.ipynb                      # email graph: stats, spy plot, assortativity, centrality, influence maximisation
│   ├── community_detection_measures_email-EU.ipynb   # email graph: community detection vs department labels
│   └── analysis_twitch.ipynb                  # Twitch graph: stats, attributes, centrality, communities, influence maximisation
└── graph_ml/                                  # part 2: graph ML
    ├── embeddings_notebook.ipynb              # email graph: spectral / Node2Vec embeddings, GAT
    └── twitch_tasks_notebook.ipynb            # Twitch graph: attribute prediction, link prediction, GraphSAGE
```

`utils.py` holds the data download and loading code, plotting helpers (static matplotlib and interactive Altair), centrality and community-quality metrics, the community detection grid search, and the influence maximisation algorithms. The two `graph_ml` notebooks are self-contained: `embeddings_notebook.ipynb` carries its own copy of the helper code in its second cell (keep it collapsed when reading).

## Part 1: network analysis

**Structure and centrality.** Degree and clique-size distributions, diameter, clustering and path lengths for both graphs. On the email graph, a spy plot of the adjacency matrix ordered by department and per-department assortativity show which departments mostly email among themselves. PageRank, betweenness, closeness, Laplacian and eigenvector centrality are compared by rank. On Twitch, degree, closeness, eigenvector and betweenness centrality are correlated with the user attributes.

**Community detection.** Louvain, Leiden and spectral clustering, each tuned with a grid search (resolution 0.2–5.0 for Louvain and Leiden, number of clusters *k* for spectral). Partitions are ranked with a structural score that none of the algorithms optimises directly:

$$\text{Composite} = 0.4\,(1 - \text{conductance}) + 0.3\,\text{internal density} + 0.3\,\text{coverage}$$

On the email graph the partitions are also scored against the departments with the adjusted Rand index (ARI). The best-scoring partitions are coarse (about 6 communities, ARI ≈ 0.20). Requiring at least 40 communities lowers the composite score but raises the ARI to 0.62 for Leiden, against 0.43 for spectral clustering. On Twitch, the communities are profiled by share of mature and partner channels, views and account age.

**Influence maximisation.** Greedy seed selection (CELF is also implemented) under the independent cascade model with Monte Carlo simulation. The email scenario is a phishing attack spreading from compromised accounts; the Twitch scenario is choosing streamers to sponsor a game. The email notebook also compares infection rate per department with that department's assortativity.

## Part 2: graph machine learning

**Email graph** (`embeddings_notebook.ipynb`): predict each member's department and predict held-out edges.

- Spectral embedding (Laplacian eigenmaps) vs Node2Vec, with a *p*, *q* grid search for Node2Vec.
- Several classifier heads on each embedding (logistic regression, SVM with RBF kernel, random forest, MLP) to test how linearly separable the embeddings are.
- Link prediction with 10% of edges hidden, embeddings retrained on the rest.
- GAT for node classification, and GAT encoder with dot-product decoder for link prediction. The graph has no node features, so the GAT gets identity features.

**Twitch graph** (`twitch_tasks_notebook.ipynb`):

- **Task A1:** predict `is_mature` and `twitch_partner` from spectral and Node2Vec embeddings with SVM, random forest, MLP, XGBoost and LightGBM heads, against a random baseline. Reported with accuracy and macro-F1, since both targets are imbalanced.
- **Task A2:** link prediction from embedding similarity only.
- **Task B:** link prediction with a GraphSAGE encoder and MLP edge scorer written in PyTorch. Train/validation/test edges are split before any features are computed, so held-out edges never take part in message passing. Node features are structural (degree, clustering, core number, PageRank), and training stops early on validation average precision.

## Results

Numbers are on the test split of a single run with seed 42.

| Graph | Task | Model | Result |
|---|---|---|---|
| Email | Department classification | Spectral embedding | accuracy 0.11 |
| Email | Department classification | Node2Vec (*p* = 1, *q* = 1) | accuracy 0.83 |
| Email | Department classification | Node2Vec (*p* = 1, *q* = 4) | accuracy 0.87 |
| Email | Link prediction | Spectral embedding | ROC-AUC 0.85 |
| Email | Link prediction | Node2Vec (*p* = 1, *q* = 1) | ROC-AUC 0.88 |
| Email | Link prediction | GAT | ROC-AUC ≈ 0.78 |
| Twitch | `is_mature` | Node2Vec + SVM (RBF) | macro-F1 0.640, accuracy 0.676 |
| Twitch | `twitch_partner` | Node2Vec + SVM (RBF) | macro-F1 0.813, accuracy 0.909 |
| Twitch | Link prediction | Spectral embedding similarity | ROC-AUC 0.790, AP 0.779 |
| Twitch | Link prediction | Node2Vec embedding similarity | ROC-AUC 0.592, AP 0.580 |
| Twitch | Link prediction | GraphSAGE + MLP scorer | ROC-AUC 0.908, AP 0.909, accuracy 0.823 |

On the email graph Node2Vec beat GAT on both tasks; the notebook discusses why (no node attributes, small graph).

## Setup

```bash
python -m venv .graph_venv
# Windows: .graph_venv\Scripts\activate    macOS/Linux: source .graph_venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn node2vec xgboost lightgbm tabulate   # used by the notebooks, not yet in requirements.txt
```

Download the data into `data/` (git-ignored) by running this from the repository root:

```python
from utils import download_data

download_data("email", "https://snap.stanford.edu/data/",
              ["email-Eu-core.txt.gz", "email-Eu-core-department-labels.txt.gz"], category="email")
download_data("twitch", "https://github.com/Carda01/community_detection/releases/download/1.0/",
              ["twitch_data.zip"], category="twitch")
```

The loaders in `utils.py` read from `data/...` relative to the working directory, so run the notebooks with the repository root as the working directory (in VS Code: set `jupyter.notebookFileRoot` to `${workspaceFolder}`).

## Authors

- Kristóf Balázs ([@Kiklar](https://github.com/Kiklar))
- Stefanos Kypritidis ([@stef4k](https://github.com/stef4k))
- Alfio Cardillo [@Carda01](https://github.com/Carda01)
