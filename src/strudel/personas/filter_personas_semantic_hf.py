import argparse
import json
import os
import random
from statistics import mean

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from strudel.util import load_json, load_jsonl, save_json, save_jsonl

random.seed(42)

domains = [
    "Ecology",
    "Evolutionary Biology",
    "Urban Planning",
    "Religion",
    "Linguistics",
    "Manufacturing",
    "Blockchain",
    "Healthcare",
    "Wildlife Conservation",
    "Mathematics",
    "Environmental Science",
    "Psychology",
    "Nanotechnology",
    "Politics",
    "Neuroscience",
    "Human Resources",
    "Education",
    "Dance",
    "Diseases",
    "Architecture",
    "Archaeology",
    "Agriculture",
    "Astronomy",
    "Artificial Intelligence",
    "Biology",
    "Music",
    "Language",
    "Energy",
    "Environmental Science",
    "Finance",
    "Earth Science",
    "Biomedical Research",
    "Economics",
    "Wildlife Biology",
    "Woodworking",
    "Telecommunications",
    "Transportation",
    "Sports",
    "Adventure",
    "Sustainability",
    "Medicine",
    "Electronics",
    "Food",
    "Computer Science",
    "Outdoors",
    "Physics",
    "Nuclear",
    "Diversity and Inclusion",
    "Social Science",
    "Climate Change",
    "History",
    "Ethics",
    "Geography",
    "Automation Technology",
    "Aviation",
    "Photography",
    "Space Exploration",
    "Philosophy",
    "Materials Science",
    "Statistics",
    "Logistics",
    "Teaching",
    "Optics",
    "Literacy",
    "Fashion",
    "Cybersecurity",
    "Culture",
    "Animals",
    "Arts and Crafts",
    "Geology",
]


class PersonaSearch:
    def __init__(
        self,
        output_dir: str,
        input_path: str,
        query_path: str = None,
        embedding_model="all-MiniLM-L6-v2",
        index_type="Flat",
        top_k=100,
        index_path="/data/data/complex_images/personas/index.faiss",
    ):
        self.output_dir = output_dir
        self.input_path = input_path
        self.embedding_model = SentenceTransformer(model_name_or_path=embedding_model, device="cuda", trust_remote_code=True)
        self.index_type = index_type
        self.top_k = top_k
        self.index_path = index_path
        self.index = None
        self.dataset = None
        self.categories = load_json(query_path) if query_path else None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _load_index(self):
        if self.dataset is None:
            if not self.input_path or not os.path.exists(self.index_path):
                raise RuntimeError("Dataset or FAISS index not found. Run `.ingest()` first.")
            self.dataset = load_dataset("json", data_files=self.input_path, split="train")
            self.dataset.load_faiss_index("embedding", self.index_path)

    def _embed_batch(self, batch, field: str = None, input_batch_size: int = 32):
        embs = self.embedding_model.encode(
            batch[field] if field else batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=input_batch_size,
            show_progress_bar=False,
        ).astype(np.float32)
        return {"embedding": [e for e in embs]}

    def ingest(self, input_batch_size: int = 32, train_size: int = 10**5):
        self.dataset = load_dataset("json", data_files=self.input_path, split="train")
        self.dataset = self.dataset.map(
            lambda batch: self._embed_batch(batch, field="persona", input_batch_size=input_batch_size),
            batched=True,
            batch_size=input_batch_size,
            writer_batch_size=10_000,
            keep_in_memory=False,
            load_from_cache_file=False,
        )
        self.dataset.to_json(f"{self.output_dir}/embeddings.jsonl")

        self.dataset = self.dataset.map(lambda example: {"embedding": np.array(example["embedding"], dtype=np.float32)})
        self.dataset.add_faiss_index(
            column="embedding",
            device=-1,
            string_factory=self.index_type,
            metric_type=faiss.METRIC_INNER_PRODUCT,
            train_size=train_size,
        )
        self.dataset.save_faiss_index(index_name="embedding", file=self.index_path)

        print("Embeddings and index saved.")

    def search(self, query):
        self._load_index()
        query_emb = self._embed_batch([query])["embedding"][0]
        scores, retrieved_samples = self.dataset.get_nearest_examples("embedding", np.array(query_emb, dtype=np.float32), k=self.top_k)
        return [{"id": s["id"], "persona": s["persona"], "score": float(scores[i])} for i, s in enumerate(retrieved_samples)]

    def search_batch(self, queries, score_threshold=None):
        """
        Search personas for multiple queries at once.
        :param queries: list of query dicts
        Returns:
            results_by_query: {query_text: [ {id, text, score}, ...], ...}
            deduped_personas: [ {id, persona, topics: [topic1, ...]}, ... ]
            topic_statistics: {topic: count_of_personas}
        """
        self._load_index()
        query_texts = [q["query"] for q in queries]
        query_topics = [q["topic"] for q in queries]

        print("Create embeddings ...")
        query_embs = self._embed_batch(query_texts)["embedding"]
        print("Get nearest neighbours ...")
        total_scores, total_examples = self.dataset.get_nearest_examples_batch(
            "embedding", np.array(query_embs, dtype=np.float32), k=self.top_k
        )

        results_by_query = {}
        deduped = {}
        topic_stats = {}

        print("Assemble data ...")
        for i, (scores, samples) in enumerate(tqdm(zip(total_scores, total_examples), total=len(total_scores))):
            topic = query_topics[i]
            query = query_texts[i]
            results = []

            if query.count(",") > 8:
                score_threshold = 0.4

            for j, score in enumerate(scores):
                if score_threshold and float(score) < score_threshold:
                    continue
                pers_id = samples["id"][j]
                pers_txt = samples["persona"][j]
                results.append({"id": pers_id, "persona": pers_txt, "score": float(score)})

                if pers_id not in deduped:
                    deduped[pers_id] = {"id": pers_id, "persona": pers_txt, "topics": {topic: 1}, "topics_scores": {topic: [float(score)]}}
                else:
                    deduped[pers_id]["topics"][topic] = deduped[pers_id]["topics"].get(topic, 0) + 1
                    if topic in deduped[pers_id]["topics_scores"]:
                        deduped[pers_id]["topics_scores"][topic].append(float(score))
                    else:
                        deduped[pers_id]["topics_scores"][topic] = [float(score)]

            results_by_query.setdefault(topic, {})[query] = results

        # Compute statistics (count how many personas belong to each topic)
        for d in deduped.values():
            for topic in d["topics"]:
                topic_stats[topic] = topic_stats.get(topic, 0) + 1

        return results_by_query, list(deduped.values()), topic_stats

    def run(self):
        queries = []
        for category, content in self.categories.items():
            if "search_queries" in content and len(content["search_queries"]) > 0:
                for query in content["search_queries"]:
                    queries.append({"topic": category, "query": query})
        print(queries)
        results_by_query, deduped_personas, topic_statistics = self.search_batch(queries=queries, score_threshold=0.45)

        for topic, results in results_by_query.items():
            save_json(filename=f"{self.output_dir}/per_query_{topic}.json", data=results)

        save_json(filename=f"{self.output_dir}/statistics.json", data=topic_statistics)
        save_json(filename=f"{self.output_dir}/deduped.json", data=deduped_personas)
        save_jsonl(filename=f"{self.output_dir}/deduped.jsonl", data=deduped_personas)

    def analyse_results(self, result_path, start_idx, end_idx):
        if not os.path.exists(f"{result_path}/sliced"):
            os.makedirs(f"{result_path}/sliced")
        for filename in os.listdir(result_path):
            if filename.startswith("per_query_") and filename.endswith(".json"):
                file_path = os.path.join(result_path, filename)
                data = load_json(file_path)

                sliced_data = {key: value[start_idx:end_idx] for key, value in data.items() if isinstance(value, list)}

                output_filename = filename.replace(".json", f"_sliced_{start_idx}_{end_idx}.json".replace("-", "m"))
                output_dir = os.path.join(f"{result_path}/sliced", output_filename)
                save_json(output_dir, sliced_data)

    def post_process(self) -> str:
        print("Post-Processing ...")

        personas_load = load_jsonl(filename=f"{self.output_dir}/deduped.jsonl")
        random.shuffle(personas_load)

        for item in personas_load:
            item["mean_score"] = {k: mean(v) for k, v in item["topics_scores"].items()}

        personas = {}
        statistics = {}
        statistics_minor = {}
        with open(f"{self.output_dir}/per_category.jsonl", "a", encoding="utf-8") as f:
            with open(f"{self.output_dir}/per_category_minor.jsonl", "a", encoding="utf-8") as fm:
                for category, content in tqdm(self.categories.items()):
                    category_search = category
                    if category in ["sequence", "activity", "component", "usecase", "state"]:
                        category_search = "class"
                    elif category in ["gantt"]:
                        category_search = "bpmn"
                    assign_domain = content["group"] == "modeling" and category not in ["bpmn", "gantt", "requirement"]

                    samples_per_category = 50000
                    min_matches = 2
                    min_thresh = 0.65
                    if category in ["mol", "music"]:
                        min_thresh = 0.7
                    if assign_domain:
                        samples_per_category = 5000
                        min_matches = 4
                        min_thresh = 0.7

                    candidates = [
                        p
                        for p in personas_load
                        if category_search in p["topics"]
                        and (
                            p["topics"][category_search] >= min_matches
                            or p["mean_score"][category_search] >= min_thresh
                            or (category_search == "dna" and (" dna " in p["persona"].lower() or "genetics" in p["persona"].lower()))
                        )
                    ]
                    candidates.sort(key=lambda p: p["mean_score"][category_search], reverse=True)

                    cnt = min(samples_per_category, len(candidates))
                    personas_category = []
                    idx = 0
                    for p in candidates[:cnt]:
                        for dom in random.sample(domains, 10):
                            idx += 1
                            personas_category.append(
                                {
                                    "id": p["id"],
                                    "idx": f"{category}_{idx}",
                                    "category": category,
                                    "persona": p["persona"],
                                    "scores": {
                                        "matches": p["topics"][category_search],
                                        "mean_score": p["mean_score"][category_search],
                                        "scores": p["topics_scores"][category_search],
                                    },
                                    "domain": dom if assign_domain else "",
                                }
                            )
                            if not assign_domain:
                                break

                    personas[category] = personas_category
                    statistics[category] = len(personas_category)
                    for pc in personas_category:
                        f.write(f"{json.dumps(pc, ensure_ascii=False)}\n")

                    # also save samples with just 1 match
                    candidates_minor = candidates[cnt:] + [
                        p
                        for p in personas_load
                        if category_search in p["topics"]
                        and (p["topics"][category_search] < min_matches and p["mean_score"][category_search] < min_thresh)
                    ]
                    candidates_minor.sort(key=lambda p: p["mean_score"][category_search], reverse=True)
                    personas_category_minor = [
                        {
                            "id": f"min_{p['id']}",
                            "idx": f"min_{category}_{i+1}",
                            "category": category,
                            "persona": p["persona"],
                            "scores": {
                                "matches": p["topics"][category_search],
                                "mean_score": p["mean_score"][category_search],
                                "scores": p["topics_scores"][category_search],
                            },
                            "domain": domains[random.randint(0, 69)] if assign_domain else "",
                        }
                        for i, p in enumerate(candidates_minor)
                    ]
                    statistics_minor[category] = len(personas_category_minor)
                    for pcm in personas_category_minor:
                        fm.write(f"{json.dumps(pcm, ensure_ascii=False)}\n")

        save_json(filename=f"{self.output_dir}/per_category.json", data=personas)
        save_json(filename=f"{self.output_dir}/statistics_post.json", data=statistics)
        save_json(filename=f"{self.output_dir}/statistics_post_minor.json", data=statistics_minor)
        category_dir = os.path.join(self.output_dir, "per_category")
        os.makedirs(category_dir)
        for category, items in personas.items():
            output_path = os.path.join(category_dir, f"{category}.json")
            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(items, outfile, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run persona semantic search.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--query-path", required=False, help="Path to search queries")
    parser.add_argument("--index-type", type=str, default="Flat", help="Index type")
    parser.add_argument("--embedding-model", type=str, default="Snowflake/snowflake-arctic-embed-l-v2.0", help="Embedding Model")

    args = parser.parse_args()

    searcher = PersonaSearch(
        output_dir=args.output,
        input_path=args.input,
        query_path=args.query_path,
        embedding_model=args.embedding_model,
        index_type=args.index_type,
        top_k=50000,
        index_path="/data/personas/snow_index_full_flat.faiss",
    )

    searcher.run()
    searcher.post_process()


if __name__ == "__main__":
    main()
