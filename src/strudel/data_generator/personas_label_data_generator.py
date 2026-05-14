import argparse
import json
import re
from collections import defaultdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset

from strudel.data_generator.data_generator import DataGenerator
from strudel.prompt_templates.templates_personas import PERSONA_LABEL_PROMPT, PERSONA_LABEL_SYSTEM
from strudel.util import check_reasoning


def create_figure(inputs, cluster_summaries: Dict[int, str], output_path: str) -> None:
    """Creates a figure of the clustered texts and save it as an artifact.
    Args:
        inputs: The inputs of the step, as we will extract information from them again.
        cluster_summaries: The summaries of the clusters, obtained from the LLM.
    """
    print("🖼️ Creating figure for the clusters...")

    label2docs = defaultdict(list)
    for i, label in enumerate(inputs["cluster_label"]):
        label2docs[label].append(i)

    labels = []
    projections = []
    id2cluster = {}
    for i, input in enumerate(inputs):
        label = input["cluster_label"]
        id2cluster[i] = label
        labels.append(label)
        projections.append(input["projection"])

    projections = np.array(projections)

    # Contains the placement of the cluster centers in the figure
    cluster_centers: Dict[str, Tuple[float, float]] = {}
    for label in label2docs.keys():
        x = np.mean([projections[doc, 0] for doc in label2docs[label]])
        y = np.mean([projections[doc, 1] for doc in label2docs[label]])
        cluster_centers[label] = (x, y)

    df = pd.DataFrame(
        data={
            "X": projections[:, 0],
            "Y": projections[:, 1],
            "labels": labels,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    unique_labels = df["labels"].unique()
    # Map of colors for each label (-1 is black)
    colormap = {label: color for label, color in zip(unique_labels, plt.cm.Spectral(np.linspace(0, 1, len(unique_labels))))}
    colormap[-1] = np.array([0, 0, 0, 0])
    df["color"] = df["labels"].map(colormap)

    df.plot(
        kind="scatter",
        x="X",
        y="Y",
        c="color",
        s=0.75,
        alpha=0.8,
        linewidth=0.4,
        ax=ax,
        colorbar=False,
    )

    for label in cluster_summaries.keys():
        if label == -1:
            continue
        summary = str(cluster_summaries[label])
        position = cluster_centers[label]
        t = ax.text(
            position[0],
            position[1],
            summary,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=4,
        )
        t.set_bbox(dict(facecolor="white", alpha=0.9, linewidth=0, boxstyle="square,pad=0.1"))

    ax.set_axis_off()

    fig.savefig(output_path)
    plt.close()


class StructuredDataGenerator(DataGenerator):
    SYSTEM_PROMPT = PERSONA_LABEL_SYSTEM
    USER_TEMPLATE = PERSONA_LABEL_PROMPT

    def _setup_dataset(self):
        samples_per_cluster = 10

        dataset_load = load_dataset("json", data_files=self.input_path, split="train")
        labels = dataset_load["cluster_label"]

        label2docs = defaultdict(list)
        for i, label in enumerate(labels):
            label2docs[label].append(i)

        input_labels = []
        input_texts = []
        for label in set(labels):
            if label != -1:
                # Get the ids but remove possible duplicates
                ids = set(np.random.choice(label2docs[label], size=samples_per_cluster))
                examples = [dataset_load["persona"][int(i)] for i in ids]
                input_texts.append("\n\n".join([f"Example {i}:\n{t}" for i, t in enumerate(examples, start=1)]))
                input_labels.append(label)

        self.dataset = Dataset.from_dict({"label": input_labels, "text": input_texts})

    def get_system_prompt(self, sample: dict) -> str:
        return self.SYSTEM_PROMPT

    def format_prompt(self, sample: dict) -> str:
        return self.USER_TEMPLATE.format(personas=sample["text"])

    def format_output(self, sample: dict, response: str, finish_reason: str) -> dict:
        match = re.search(r"\{.*\}", check_reasoning(response), re.DOTALL)
        try:
            labels_json = json.loads(match.group(0)) if match else {}
        except:
            labels_json = {}
        if "labels" not in labels_json or len(labels_json["labels"]) < 1:
            print(f"NO LABELS: {response}")
        return {
            "input": sample["original"],
            "prompt": sample["prompt"],
            "generation": response.strip(),
            "labels": labels_json["labels"] if "labels" in labels_json else [],
            "finish_reason": finish_reason,
        }

    def post_process(self, file_path_gen: str = None, file_path_pers: str = None) -> str:
        default_label = ["None"]

        data_path_gen = file_path_gen if file_path_gen else self.output_path
        data_path_pers = file_path_pers if file_path_pers else self.input_path
        ds_generated = load_dataset("json", data_files=data_path_gen, split="train")
        ds_personas = load_dataset("json", data_files=data_path_pers, split="train")

        cluster_summaries: Dict[int, str] = {-1: default_label}
        for sample in ds_generated:
            cluster_summaries[sample["input"]["label"]] = [str(x) for x in sample["labels"]]

        ds_personas = ds_personas.map(lambda sample: {"summary_label": cluster_summaries[sample["cluster_label"]]})
        ds_personas.to_json(f"{data_path_gen.split('.')[0]}_post.jsonl")

        create_figure(ds_personas, cluster_summaries, f"{data_path_gen.split('.')[0]}_figure.png")


def main():
    parser = argparse.ArgumentParser(description="Run persona extraction with vLLM.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--data-batch-size", type=int, default=1000, help="Batch size per LLM call")
    parser.add_argument("--start-index", type=int, default=0, help="Start index within the dataset")

    args = parser.parse_args()

    generator = StructuredDataGenerator(
        model_config=args.config,
        output_path=args.output,
        input_path="/models/envs/venv-distil/personas_clustered_100k.jsonl",
        max_samples=args.max_samples,
        data_batch_size=args.data_batch_size,
        start_index=args.start_index,
    )

    generator.run()
    generator.post_process()


if __name__ == "__main__":
    main()

