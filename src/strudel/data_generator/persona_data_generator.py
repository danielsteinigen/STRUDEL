import argparse

from datasets import load_dataset

from strudel.data_generator.data_generator import DataGenerator
from strudel.prompt_templates.templates_personas import TEXT_TO_PERSONA_PROMPT, TEXT_TO_PERSONA_SYSTEM
from strudel.util import check_reasoning


class PersonaDataGenerator(DataGenerator):
    SYSTEM_PROMPT = TEXT_TO_PERSONA_SYSTEM
    USER_TEMPLATE = TEXT_TO_PERSONA_PROMPT

    def _setup_dataset(self):
        self.dataset = load_dataset(
            path="HuggingFaceFW/fineweb-edu", name="sample-10BT", revision="v1.4.0", split="train", streaming=True  # CC-MAIN-2025-26
        )

    def get_system_prompt(self, sample: dict) -> str:
        return self.SYSTEM_PROMPT

    def format_prompt(self, sample: dict) -> str:
        text = sample.get("text", "").strip()
        if not text:
            print("No 'text' field")
            return None
        text_trunc = self._truncate(text=text, max_length=4000)
        return self.USER_TEMPLATE.format(text=text_trunc)

    def format_output(self, sample: dict, response: str, finish_reason: str) -> dict:
        return {"id": sample["original"].get("id", ""), "persona": response.strip(), "finish_reason": finish_reason}

    def post_process(self, file_path: str = None) -> str:
        data_path = file_path if file_path else self.output_path
        ds = load_dataset(
            "json",
            data_files=data_path,
            split="train",
        )
        ds = ds.filter(lambda sample: sample["finish_reason"] == "stop")
        ds = ds.remove_columns(["finish_reason"])
        ds = ds.map(lambda sample: {"persona": check_reasoning(sample["persona"])})
        ds.to_json(f"{data_path.split('.')[0]}_post.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Run persona extraction with vLLM.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--data-batch-size", type=int, default=1000, help="Batch size per LLM call")
    parser.add_argument("--start-index", type=int, default=None, help="Start index within the dataset")

    args = parser.parse_args()

    generator = PersonaDataGenerator(
        model_config=args.config,
        output_path=args.output,
        max_samples=args.max_samples,
        data_batch_size=args.data_batch_size,
        start_index=args.start_index,
    )

    generator.run()
    generator.post_process()


if __name__ == "__main__":
    main()

# python src/data_generator/persona_data_generator.py --config src/data_generator/configs/smollm3-3b.yaml --output /data/data/complex_images/personas/personas.jsonl --max-samples 500 --data-batch-size 100
