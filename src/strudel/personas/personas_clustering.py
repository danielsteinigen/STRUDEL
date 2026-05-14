from collections import Counter

import faiss
import joblib
import numpy as np
from datasets import load_dataset
from sklearn.cluster import DBSCAN
from umap import UMAP

if __name__ == "__main__":

    ds = load_dataset("json", data_files="/data/personas/personas_embed.jsonl", split="train")
    ds = ds.map(lambda example: {"embedding": np.array(example["embedding"], dtype=np.float32)})

    ds.add_faiss_index(
        column="embedding",
        device=-1,
        string_factory="IVF2500,Flat",  # "Flat"
        metric_type=faiss.METRIC_INNER_PRODUCT,
        train_size=10**5,
    )
    ds.save_faiss_index(index_name="embedding", file="/models/envs/venv-distil/qwen_index.faiss")

    print("🏋️‍♀️ Start UMAP training...")
    umap = UMAP(
        n_components=2,  # The dimension of the space to embed into (2 to 100)
        metric="cosine",  # The metric to use to compute distances in high dimensional space.
        n_jobs=2,  # The number of parallel jobs to run
        random_state=42,  # The random state to use for the UMAP algorithm
    )
    mapper = umap.fit(ds["embedding"])
    print("🏅 UMAP training done!")

    ds.drop_index("embedding")
    ds = ds.remove_columns(["embedding"])
    ds = ds.map(lambda sample, idx: {"projection": mapper.embedding_[idx]}, with_indices=True)
    ds.to_json(f"/models/envs/venv-distil/test_projection.jsonl")

    with open(str("/models/envs/venv-distil/qwen_umap.joblib"), "wb") as f:
        joblib.dump(mapper, f)

    print("🏋️‍♀️ Start training DBSCAN...")
    clusterer = DBSCAN(
        eps=0.11,  # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples=20,  # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
        metric="euclidean",  # The metric to use when calculating distance between instances in a feature array
        n_jobs=8,  # The number of parallel jobs to run.
    )
    fitted_clusterer = clusterer.fit(ds["projection"])
    print("🏅 DBSCAN training done!")
    print(f"DBSCAN labels assigned: {len(set(fitted_clusterer.labels_))}")

    ds = ds.map(lambda sample, idx: {"cluster_id": fitted_clusterer.labels_[idx]}, with_indices=True)  # -1 means it wasn't clustered
    ds.to_json(f"/models/envs/venv-distil/test_projection.jsonl")

    with open(str("/models/envs/venv-distil/qwen_dbscan.joblib"), "wb") as f:
        joblib.dump(fitted_clusterer, f)

    print(Counter(ds["cluster_id"]))
