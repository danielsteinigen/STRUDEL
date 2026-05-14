"""
Pipeline runner with CLI args.

Usage:
    python persona_pipeline.py --model_config src/data_generator/configs/qwen-3-235b-instruct.yaml
"""

import argparse

from strudel.data_generator.persona_data_generator import PersonaDataGenerator
from strudel.data_generator.persona_query_data_generator import PersonaQueryDataGenerator
from strudel.personas.filter_personas_semantic_hf import PersonaSearch

STEPS = [
    (
        PersonaQueryDataGenerator,
        dict(
            model_config="src/data_generator/configs/qwen-3-235b-instruct.yaml",
            output_path="data/personas/personas_query.jsonl",
            input_path="data/categories_all.json",
        ),
    ),
    (
        PersonaDataGenerator,
        dict(
            model_config="src/data_generator/configs/qwen-3-235b-instruct.yaml",
            output_path="data/personas/personas.jsonl",
            data_batch_size=60000,
        ),
    ),
    (
        PersonaSearch,
        dict(
            input_path="data/personas/personas_post.jsonl",
            output_path="data/personas/personas_filtered",
            query_path="data/personas/personas_query_post.json",
            top_k=10000,
            embedding_model="all-MiniLM-L6-v2",
        ),
    ),
]


def parse_args():
    p = argparse.ArgumentParser(description="Run persona pipeline.")
    p.add_argument("--model-config", type=str, help="Path to model config file.")
    p.add_argument("--dataset_name", type=str)
    p.add_argument("--split", type=str)
    p.add_argument("--limit", type=int, help="Limit number of processed records")
    return p.parse_args()


def main():
    args = parse_args()

    for cls, kwargs in STEPS:
        # Override kwargs if CLI arg is given and the key exists in step kwargs
        for k, v in vars(args).items():
            if v is not None and k in kwargs:
                kwargs[k] = v

        print(f"\n[PIPELINE] Running {cls.__name__} with args: {kwargs}")
        step = cls(**kwargs)
        step.run()
        print(f"[PIPELINE] Post-Processing {cls.__name__}")
        step.post_process()
        print(f"[PIPELINE] Finished {cls.__name__}")


if __name__ == "__main__":
    main()
