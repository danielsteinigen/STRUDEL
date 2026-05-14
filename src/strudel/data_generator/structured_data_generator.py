import argparse
import random

from datasets import load_dataset

from strudel.data_generator.data_generator import DataGenerator
from strudel.prompt_templates.templates_generation import PROMPT_END2END_5, PROMPT_END2END_10, SYSTEM_END2END
from strudel.util import check_reasoning, extract_part, load_json

random.seed(42)

complexity = ["low", "medium", "high"]
MAX_SAMPLES = None

phrasings = [
    "",
    " The problem statement should be phrased as a question using first person singular, not starting with 'i need'.",
    " The problem statement should be phrased as a command using second person singular.",
]
chart_types_main = ["bar chart", "column chart", "line chart", "scatter plot", "pie chart"]
chart_types_side = ["stacked bar chart", "box plot", "area chart", "tree map", "spider/radar chart"]


class StructuredDataGenerator(DataGenerator):
    SYSTEM_PROMPT = SYSTEM_END2END
    USER_TEMPLATE = PROMPT_END2END_5  # PROMPT_END2END_10

    def _evolve_samples(self, batch, indices, categories):
        new_data = {
            "id": [],
            "id_compl": [],
            "category_key": [],
            "category_name": [],
            "lang_key": [],
            "lang_name": [],
            "code_instruct": [],
            "complexity": [],
            "persona": [],
            "domain": [],
        }
        cnt_samples = {cat: 0 for cat, _ in categories.items()}

        for ind, pers_id, persona_item, category_item, domain_item in zip(
            indices, batch["id"], batch["persona"], batch["category"], batch["domain"]
        ):
            cnt_samples[category_item] += 1
            if MAX_SAMPLES and cnt_samples[category_item] > MAX_SAMPLES:
                continue
            idx = 0
            id_compl = 0

            content = categories[category_item]
            for lang_key, lang in content["language"].items():
                id_compl += 1
                for compl in complexity:
                    idx += 1

                    dataset_sample = {
                        "id": f"{ind+1}_{idx}",
                        "id_compl": f"{ind+1}_{id_compl}",
                        "category_key": category_item,
                        "category_name": content["name"],
                        "lang_key": lang_key,
                        "lang_name": lang["language"],
                        "code_instruct": lang["code_instruct"],
                        "complexity": compl,
                        "persona": {"id": pers_id, "persona": persona_item},
                        "domain": domain_item if domain_item else "",
                    }

                    for key, value in dataset_sample.items():
                        new_data[key].append(value)

        return new_data

    def _setup_dataset(self):
        categories = load_json(filename="data/categories_all.json")
        self.dataset = load_dataset(
            "json",
            data_files=self.input_path,
            split="train",
        )
        self.dataset = self.dataset.map(
            self._evolve_samples,
            fn_kwargs={"categories": categories},
            batched=True,
            batch_size=10000,
            remove_columns=self.dataset.column_names,
            with_indices=True,
        )

    def get_system_prompt(self, sample: dict) -> str:
        return self.SYSTEM_PROMPT

    def format_prompt(self, sample: dict) -> str:
        category_name = (
            sample["category_name"]
            if sample["category_key"] != "chart"
            else (random.choice(chart_types_side) if random.randint(0, 2) == 0 else random.choice(chart_types_main))
        )
        category_prompt = f"{category_name} for an application in the domain {sample['domain']}" if sample["domain"] else category_name
        return self.USER_TEMPLATE.format(
            persona=sample["persona"]["persona"],
            category=category_prompt,
            lang=sample["lang_name"],
            complexity=sample["complexity"],
            code_instruct=f" {sample['code_instruct']}" if sample["code_instruct"] else "",
            phrasing=phrasings[random.randint(0, 2)],
        )

    def format_output(self, sample: dict, response: str, finish_reason: str) -> dict:
        return {"input": sample["original"], "generation": response.strip(), "finish_reason": finish_reason}

    def post_process(self, file_path: str = None, has_funct: bool = True) -> str:
        data_path = file_path if file_path else self.output_path
        ds = load_dataset(
            "json",
            data_files=data_path,
            split="train",
        )
        ds_stop = ds.filter(lambda sample: sample["finish_reason"] != "stop")
        ds = ds.filter(lambda sample: sample["finish_reason"] == "stop")
        ds = ds.remove_columns(["finish_reason"])
        ds = ds.map(lambda sample: {"generation": check_reasoning(sample["generation"])})
        ds = ds.map(lambda sample: {"code": extract_part(sample["generation"], "```", "```", True, True)})
        ds = ds.map(lambda sample: {"generation": sample["generation"].replace("**", "").strip()})
        if has_funct:
            ds = ds.map(lambda sample: {"problem": extract_part(sample["generation"], "Problem:", "Functionality:", False)})
            ds = ds.map(lambda sample: {"description": extract_part(sample["generation"], "Functionality:", "Code:", True)})
        else:
            ds = ds.map(lambda sample: {"problem": extract_part(sample["generation"], "Problem:", "Code:", False)})

        ds = ds.map(lambda sample: {"answer": extract_part(sample["generation"], "Answer:", "", True)})
        ds = ds.remove_columns(["generation"])
        ds.to_json(f"{data_path.split('.')[0]}_post.jsonl")
        ds_stop.to_json(f"{data_path.split('.')[0]}_post_stop.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Run persona extraction with vLLM.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--input", required=True, help="Path to the personas JSONL input file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--data-batch-size", type=int, default=1000, help="Batch size per LLM call")
    parser.add_argument("--start-index", type=int, default=None, help="Start index within the dataset")
    parser.add_argument("--end-index", type=int, default=None, help="End index within the dataset")
    parser.add_argument("--use-harmony", action="store_true", help="Flag indicating if harmony prompt format should be used.")

    args = parser.parse_args()

    generator = StructuredDataGenerator(
        model_config=args.config,
        output_path=args.output,
        input_path=args.input,
        max_samples=args.max_samples,
        data_batch_size=args.data_batch_size,
        start_index=args.start_index,
        end_index=args.end_index,
        use_harmony=args.use_harmony,
    )

    generator.run()
    generator.post_process()


if __name__ == "__main__":
    main()
