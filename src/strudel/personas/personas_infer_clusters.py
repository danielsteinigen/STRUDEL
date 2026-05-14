from collections import Counter

import faiss
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm

faiss_filename = "/data/personas/faiss_index/index.faiss"

print("Start dataset download...")
ds = load_dataset("json", data_files="/data/personas/personas_labeled_post.jsonl", split="train")
ds_to_infer = load_dataset("json", data_files="/data/personas/personas_embed_clean.jsonl", split="train")

print("Prepare FAISS index...")
faiss_index = faiss.read_index(faiss_filename)


def infer_clusters(row, top_k=1):
    embeddings = np.array(row["embedding"])[np.newaxis, :]
    dist, neighbours = faiss_index.search(embeddings, top_k)
    # Get the index of the neighbour we want to get the label from
    idx = int(neighbours[0][0])
    # and assign it
    row["labels"] = ds[idx]["summary_label"]
    row["cluster"] = ds[idx]["cluster_label"]
    return row


ds = ds.map(lambda sample: {"summary_label_str": str(sample["summary_label"])})
counter = Counter(ds["summary_label_str"])
print(counter)

print("Start inferring labels for the dataset...")
ds_inferred = ds_to_infer.map(infer_clusters, num_proc=16)
ds_inferred = ds_inferred.remove_columns(["model_name_embeddings", "embedding"])
ds_inferred.to_json(f"/data/personas/personas_clustered.jsonl")

print("Assembling subset of final dataset...")
samples_per_cluster = 1060
counter_assemble = {key: 0 for key in counter if "None" not in key}

ids = []
personas = []
labels = []
clusters = []
for sample in tqdm(ds):
    if "None" not in sample["summary_label"] and counter_assemble[str(sample["summary_label"])] < samples_per_cluster:
        ids.append(sample["id"])
        personas.append(sample["persona"])
        labels.append(sample["summary_label"])
        clusters.append(sample["cluster_label"])
        counter_assemble[str(sample["summary_label"])] += 1

    if all(value >= samples_per_cluster for value in counter_assemble.values()):
        break

for sample in tqdm(ds_inferred.shuffle(seed=42)):
    if "None" not in sample["labels"] and counter_assemble[str(sample["labels"])] < samples_per_cluster and sample["id"] not in ids:
        ids.append(sample["id"])
        personas.append(sample["persona"])
        labels.append(sample["labels"])
        clusters.append(sample["cluster"])
        counter_assemble[str(sample["labels"])] += 1

    if all(value >= samples_per_cluster for value in counter_assemble.values()):
        break

print(counter_assemble)

dataset_assembled = Dataset.from_dict({"id": ids, "persona": personas, "label": labels, "cluster": clusters})
dataset_assembled.shuffle(seed=42).to_json(f"/data/personas/personas_clustered_part{samples_per_cluster}.jsonl")


print("Total cluster labels:")
ds_inferred = ds_inferred.map(lambda sample: {"labels_str": str(sample["labels"])})
print(Counter(ds_inferred["labels_str"]))
