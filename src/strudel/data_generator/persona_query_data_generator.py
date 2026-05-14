import argparse
import json
import re

from datasets import Dataset, load_dataset

from strudel.data_generator.data_generator import DataGenerator
from strudel.prompt_templates.templates_personas import PERSONA_QUERY_PROMPT, PERSONA_QUERY_SYSTEM
from strudel.util import check_reasoning, load_json, save_json


class PersonaQueryDataGenerator(DataGenerator):
    SYSTEM_PROMPT = PERSONA_QUERY_SYSTEM
    USER_TEMPLATE = PERSONA_QUERY_PROMPT

    def _setup_dataset(self):
        categories = load_json(filename=self.input_path)
        samples = []
        for category, content in categories.items():
            if "terms_semantic" in content:
                samples.append({"category": category, "topic": content["name"], "terms": content["terms_semantic"]})
        self.dataset = Dataset.from_list(samples)

    def get_system_prompt(self, sample: dict) -> str:
        return self.SYSTEM_PROMPT

    def format_prompt(self, sample: dict) -> str:
        return self.USER_TEMPLATE.format(topic=sample["topic"], terms=sample["terms"])

    def format_output(self, sample: dict, response: str, finish_reason: str) -> dict:
        match = re.search(r"\{.*\}", check_reasoning(response), re.DOTALL)
        return {
            "category": sample["original"].get("category", ""),
            "prompt": sample["prompt"],
            "generation": response.strip(),
            "queries": json.loads(match.group(0)) if match else {},
            "finish_reason": finish_reason,
        }

    def post_process(self, file_path: str = None, file_path_input: str = None) -> str:
        categories = load_json(filename=file_path_input if file_path_input else self.input_path)
        data_path = file_path if file_path else self.output_path
        ds = load_dataset(
            "json",
            data_files=data_path,
            split="train",
        )
        ds = ds.filter(lambda sample: sample["finish_reason"] == "stop")
        for sample in ds:
            categories[sample["category"]]["search_queries"] = (
                sample["queries"]["queries"] if "queries" in sample["queries"] else sample["queries"]
            )

        save_json(filename=f"{data_path.split('.')[0]}_post.json", data=categories)


def main():
    parser = argparse.ArgumentParser(description="Run persona extraction with vLLM.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--input", required=True, help="Path to the categories JSON input file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--data-batch-size", type=int, default=1000, help="Batch size per LLM call")

    args = parser.parse_args()

    generator = PersonaQueryDataGenerator(
        model_config=args.config,
        output_path=args.output,
        input_path=args.input,
        max_samples=args.max_samples,
        data_batch_size=args.data_batch_size,
    )

    generator.run()
    generator.post_process()


if __name__ == "__main__":
    main()

# python src/data_generator/persona_query_data_generator.py --config src/data_generator/configs/smollm3-3b.yaml --output data/personas/personas_query.jsonl --data-batch-size 100
