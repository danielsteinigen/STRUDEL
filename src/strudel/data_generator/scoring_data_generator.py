import argparse

from datasets import load_dataset

from strudel.data_generator.data_generator import DataGenerator
from strudel.prompt_templates.templates_enrichment import PROMPT_SCORE_PS, PROMPT_SCORE_QA, SYSTEM_SCORE_PS, SYSTEM_SCORE_QA
from strudel.util import check_reasoning, extract_part

IS_QA = True


class ScoringDataGenerator(DataGenerator):
    SYSTEM_PROMPT = SYSTEM_SCORE_QA if IS_QA else SYSTEM_SCORE_PS
    USER_TEMPLATE = PROMPT_SCORE_QA if IS_QA else PROMPT_SCORE_PS

    def _setup_dataset(self):
        self.dataset = load_dataset(
            "json",
            data_files=self.input_path,
            split="train",
        )

    def get_system_prompt(self, sample: dict) -> str:
        return self.SYSTEM_PROMPT.format(category=sample["input"]["category_name"])

    def format_prompt(self, sample: dict) -> str:
        if IS_QA:
            return self.USER_TEMPLATE.format(
                language=sample["lang_name"], problem=sample["problem"], code=sample["code"], answer=sample["answer"]
            )
        else:
            return self.USER_TEMPLATE.format(
                language=sample["lang_name"], problem=sample["problem"], code=sample["code"], description=sample["description"]
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
                "score_correct": extract_part(
                    text=sample["generation"],
                    term_1="correct:",
                    term_2="explanation:",
                    return_empty=True,
                    remove_first_line=False,
                    reverse=True,
                )
            }
        )
        ds = ds.map(
            lambda sample: {
                "score_explanation": extract_part(
                    text=sample["generation"], term_1="explanation:", term_2="", return_empty=True, remove_first_line=False, reverse=True
                )
            }
        )
        ds = ds.remove_columns(["finish_reason", "generation"])
        ds.to_json(f"{data_path.split('.')[0]}_post.jsonl")
        ds_stop.to_json(f"{data_path.split('.')[0]}_post_stop.jsonl")

        ds.filter(lambda sample: sample["score_correct"].lower() == "true").to_json(f"{data_path.split('.')[0]}_post_true.jsonl")
        ds.filter(lambda sample: sample["score_correct"].lower() == "false").to_json(f"{data_path.split('.')[0]}_post_false.jsonl")
        ds.filter(lambda sample: sample["score_correct"].lower() == "null").to_json(f"{data_path.split('.')[0]}_post_null.jsonl")
        ds.filter(lambda sample: sample["score_correct"].lower() not in ["null", "true", "false"]).to_json(
            f"{data_path.split('.')[0]}_post_fail.jsonl"
        )


def main():
    parser = argparse.ArgumentParser(description="Run persona extraction with vLLM.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--input", required=True, help="Path to the dataset JSONL input file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--data-batch-size", type=int, default=1000, help="Batch size per LLM call")
    parser.add_argument("--start-index", type=int, default=None, help="Start index within the dataset")

    args = parser.parse_args()

    generator = ScoringDataGenerator(
        model_config=args.config,
        output_path=args.output,
        input_path=args.input,
        max_samples=args.max_samples,
        data_batch_size=args.data_batch_size,
        start_index=args.start_index,
        use_harmony=True,
    )

    generator.run()
    generator.post_process()


if __name__ == "__main__":
    main()

