# MovieLens Recommender Systems

An educational project that compares several recommendation methods using the
[MovieLens latest-small dataset](https://grouplens.org/datasets/movielens/latest/).

## Models

| Model | What it does | Status |
| --- | --- | --- |
| Content-based | Finds movies with similar titles, genres, and tags | Complete |
| Item-item kNN | Recommends from similarities in user ratings | Complete |
| Matrix factorization | Learns user and movie embeddings to predict ratings | Experimental |
| BPR | Learns to rank watched movies above unseen movies | Complete |
| SASRec | Uses a user's ordered history to predict the next movie | Implemented |

## Saved results

These results come from the saved notebook runs. The models use different
objectives, so their scores should not be compared directly.

| Model | Metric | Result |
| --- | --- | ---: |
| Global-mean baseline | Test RMSE | 1.044 |
| Item-item kNN | Test RMSE | 0.860 |
| Matrix factorization experiment | Test RMSE | 0.872 |
| BPR | Hit Rate@10 | 0.113 |
| BPR | NDCG@10 | 0.066 |
| BPR | MRR | 0.062 |
| SASRec | Best validation Hit@10 | **0.8185** |
| SASRec | Best validation NDCG@10 | **0.6065** |

The matrix-factorization notebook includes an SVD++-style implicit-history term,
but its saved evaluation does not use that term. Treat its RMSE as an
experimental result rather than a valid SVD++ benchmark.

## Files

- `content_based_rec.ipynb` - content-based recommendations
- `knn_rec.ipynb` - item-item collaborative filtering
- `simple_rec.ipynb` - matrix factorization experiment
- `BPR.ipynb` - Bayesian Personalized Ranking
- `sequential_recommendation/SASRec.ipynb` - sequential recommendations
- `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv` - MovieLens data

## Run the project

Create a virtual environment and install the main dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas matplotlib scikit-learn jupyterlab ipykernel
```

For SASRec, also install PyTorch and RecBole:

```powershell
python -m pip install torch recbole
```

Start JupyterLab from the repository root:

```powershell
python -m jupyter lab
```

## Dataset

This repository uses MovieLens `ml-latest-small`:

- 100,836 ratings
- 9,742 movies
- 610 users

The dataset was created by the
[GroupLens Research Group](https://grouplens.org/datasets/movielens/). Review the
MovieLens usage terms before redistributing the data or using it commercially.
