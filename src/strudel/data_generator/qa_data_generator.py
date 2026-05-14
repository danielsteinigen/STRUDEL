import argparse
import random

from datasets import load_dataset

from strudel.data_generator.data_generator import DataGenerator
from strudel.prompt_templates.templates_enrichment import PROMPT_LLM_QA, SYSTEM_LLM_QA
from strudel.util import extract_part

random.seed(42)

question_types = [
    "The question should ask for a specific label or text.",
    "The question should ask for a specific numeric value.",
    "It should be a multiple choice question with four options A, B, C, D, where the answer is the letter of the correct option.",
    'It should be a binary yes/no question, where the answer is exactly "Yes" or "No".',
]


class QaDataGenerator(DataGenerator):
    SYSTEM_PROMPT = SYSTEM_LLM_QA
    USER_TEMPLATE = PROMPT_LLM_QA

    def _setup_dataset(self):
        self.dataset = load_dataset(
            "json",
            data_files=self.input_path,
            split="train",
        )
        self.dataset.map(lambda sample: {"llm_question_type": random.choice(question_types)})

    def get_system_prompt(self, sample: dict) -> str:
        return self.SYSTEM_PROMPT

    def format_prompt(self, sample: dict) -> str:
        return self.USER_TEMPLATE.format(
            category=sample["category_name"], language=sample["lang_name"], code=sample["code"], question_type=sample["llm_question_type"]
        )

    def format_output(self, sample: dict, response: str, finish_reason: str) -> dict:
        new_sample = sample["original"]
        new_sample["generation"] = response.strip()
        new_sample["finish_reason"] = finish_reason
        return new_sample

    def post_process(self, file_path: str = None) -> str:
        data_path = file_path if file_path else self.output_path
        ds = load_dataset(
            "json",
            data_files=data_path,
            split="train",
        )
        ds_stop = ds.filter(lambda sample: sample["finish_reason"] != "stop")
        ds = ds.filter(lambda sample: sample["finish_reason"] == "stop")
        ds = ds.map(
            lambda sample: {
                "llm_user": extract_part(
                    text=sample["generation"], term_1="User:", term_2="Assistant:", return_empty=True, remove_first_line=False, reverse=True
                )
            }
        )
        ds = ds.map(
            lambda sample: {
                "llm_assistant": extract_part(
                    text=sample["generation"], term_1="Assistant:", term_2="", return_empty=True, remove_first_line=False, reverse=True
                )
            }
        )
        ds = ds.remove_columns(["finish_reason", "generation"])
        ds.to_json(f"{data_path.split('.')[0]}_post.jsonl")
        ds_stop.to_json(f"{data_path.split('.')[0]}_post_stop.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Run persona extraction with vLLM.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--input", required=True, help="Path to the dataset JSONL input file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--data-batch-size", type=int, default=1000, help="Batch size per LLM call")
    parser.add_argument("--start-index", type=int, default=None, help="Start index within the dataset")

    args = parser.parse_args()

    generator = QaDataGenerator(
        model_config=args.config,
        output_path=args.output,
        input_path=args.input,
        max_samples=args.max_samples,
        data_batch_size=args.data_batch_size,
        start_index=args.start_index,
    )

    generator.run()
    generator.post_process()


if __name__ == "__main__":
    main()

