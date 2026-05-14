from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from statistics import mean

from strudel.util import check_dirs, check_reasoning, extract_part, load_json, save_json, save_text

STRUCTIVIZE_RENDERER_MODULES = [
    "structivize.renderers.biology",
    "structivize.renderers.business",
    "structivize.renderers.charts",
    "structivize.renderers.chemistry",
    "structivize.renderers.culture",
    "structivize.renderers.datastructure",
    "structivize.renderers.drawing",
    "structivize.renderers.electronics",
    "structivize.renderers.modeling",
]

MAP_ANSWER = {
    "North Macedonia": "Macedonia",
    "United Arab Emirates": "Arab",
    "Central African Republic": "Africa",
    "Bosnia and Herzegovina": "Bosnia",
    "Sri Lanka": "Lanka",
    "Papua New Guinea": "Guinea",
    "South Africa": "South",
    "Saudi Arabia": "Arabia",
    "Dominican Republic": "Dominican",
    "Sierra Leone": "Sierra",
    "United Kingdom": "Kingdom",
    "Trinidad and Tobago": "Trinidad",
    "South Korea": "Korea",
    "Costa Rica": "Rica",
    "United States of America": "America",
}

TASK_MAP = {
    "code": "code",
    "analytical": "analytical",
    "structural_count": "quantification",
    "identification": "identification",
    "association": "association",
    "consistency": "association",
    "structural": "analytical",
    "quantification": "quantification",
}


class NullProgress:
    def __init__(self, total: int = 0, desc: str | None = None):
        self.total = total
        self.desc = desc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n: int = 1) -> None:
        return None


def create_progress(*, total: int, desc: str):
    try:
        tqdm_module = importlib.import_module("tqdm")
        return tqdm_module.tqdm(total=total, desc=desc)
    except ImportError:
        return NullProgress(total=total, desc=desc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and score STRUDEL code-translation outputs from the JSON files produced by 'strudel eval strudel'."
    )
    parser.add_argument("--input-files", nargs="+", required=True, help="Prediction JSON files produced by 'strudel eval strudel'.")
    parser.add_argument("--output-dir", required=True, help="Output directory for rendered files and aggregated results.")
    parser.add_argument(
        "--categories",
        default="diagram_categories.json",
        help="Path to the category definition JSON that contains renderer mappings.",
    )
    parser.add_argument("--gold-statistics", required=True, help="Path to the gold statistics JSON keyed by sample id.")
    parser.add_argument("--max-workers", type=int, default=min(multiprocessing.cpu_count(), 8), help="Parallel render workers.")
    parser.add_argument("--wandb-project", default=None, help="Optional Weights & Biases project name for logging.")
    return parser.parse_args()


@lru_cache(maxsize=None)
def get_renderer_class():
    for module_name in STRUCTIVIZE_RENDERER_MODULES:
        importlib.import_module(module_name)

    renderer_module = importlib.import_module("structivize.renderer")
    return renderer_module.Renderer


def extract_code_answer(sample: dict) -> str:
    result = sample.setdefault("result", {})
    answer = result.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    out_no_reas = check_reasoning(result.get("generation", ""))
    out_no_reas = out_no_reas.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    answer = extract_part(out_no_reas, "```", "```", False, True).strip()
    result["answer"] = answer
    result["generation_split"] = out_no_reas
    return answer


def recompute_non_code_correct(sample: dict) -> bool:
    result = sample.setdefault("result", {})
    if "correct" in result:
        return bool(result["correct"])

    out_no_reas = check_reasoning(result.get("generation", ""))
    out_no_reas = out_no_reas.replace("*", "").replace("boxed", "")
    answer_gold = str(sample.get("answer", "")).strip()
    if sample.get("category_key") == "geo" and answer_gold in MAP_ANSWER:
        answer_gold = MAP_ANSWER[answer_gold]

    num_chars = len(answer_gold) + 15
    answers = [out_no_reas[:num_chars]]
    if num_chars < len(out_no_reas):
        answers.append(out_no_reas[len(out_no_reas) - num_chars :])
        if "nswer:" in out_no_reas:
            answers.append(out_no_reas.split("nswer:", 1)[1][:num_chars])
        if "answer is" in out_no_reas:
            answers.append(out_no_reas.split("answer is", 1)[1][:num_chars])

    correct = False
    for answ in answers:
        check_gold = answer_gold
        check_pred = answ
        if check_gold not in ["A", "B", "C", "D"]:
            check_gold = answer_gold.lower()
            check_pred = answ.lower()

        if check_gold in check_pred:
            correct = True
            break

        if check_gold in ["yes", "no"]:
            check_gold = check_gold.replace("yes", "true").replace("no", "false")
            if check_gold in check_pred:
                correct = True
                break

    result["answer"] = answers
    result["correct"] = correct
    result["generation_split"] = out_no_reas
    return correct


def compute_statistics(data: list[dict], model_name: str, wandb_project: str | None = None) -> dict:
    difficulty_stats = defaultdict(list)
    cat_stats = defaultdict(list)
    domain_stats = defaultdict(list)
    type_stats = defaultdict(list)
    task_type_stats = defaultdict(list)
    task_comb_stats = defaultdict(list)

    for sample in data:
        correct = sample["result"]["correct"]
        difficulty_stats[sample["difficulty"]].append(correct)
        cat_stats[sample["category_key"]].append(correct)
        domain = sample["domain"] if sample.get("domain") else "business" if sample["category_key"] in ["table", "chart"] else "None"
        domain_stats[domain].append(correct)
        type_stats[sample["type"]].append(correct)
        task_type_stats[sample["task_type"]].append(correct)
        task_comb_stats[TASK_MAP[sample["task_type"]]].append(correct)

    correct_flags = [sample["result"]["correct"] for sample in data]
    code_samples = [sample for sample in data if sample["type"] == "code"]
    correct_code = [sample["result"]["correct"] for sample in code_samples]
    correct_code_stat_score = [sample["result"].get("correct_stat_score", 0.0) for sample in code_samples]
    correct_code_stat_f1 = [sample["result"].get("correct_stat_f1", 0.0) for sample in code_samples]
    correct_code_render = [sample["result"].get("correct_render", 0.0) for sample in code_samples]

    stats = {
        "total": {
            "samples": len(data),
            "correct_count": sum(correct_flags),
            "correct_mean": mean(correct_flags) if correct_flags else 0,
        },
        "code": {
            "correct_code": mean(correct_code) if correct_code else 0,
            "correct_code_render": mean(correct_code_render) if correct_code_render else 0,
            "correct_code_stat_f1": mean(correct_code_stat_f1) if correct_code_stat_f1 else 0,
            "correct_code_stat_score": mean(correct_code_stat_score) if correct_code_stat_score else 0,
        },
        "task_combined": {
            key: {"total": len(values), "count": sum(values), "mean": mean(values) if values else 0}
            for key, values in task_comb_stats.items()
        },
        "task_type": {
            key: {"total": len(values), "count": sum(values), "mean": mean(values) if values else 0}
            for key, values in task_type_stats.items()
        },
        "per_type": {
            key: {"total": len(values), "count": sum(values), "mean": mean(values) if values else 0}
            for key, values in type_stats.items()
        },
        "per_category": {
            key: {"total": len(values), "count": sum(values), "mean": mean(values) if values else 0}
            for key, values in cat_stats.items()
        },
        "per_domain": {
            key: {"total": len(values), "count": sum(values), "mean": mean(values) if values else 0}
            for key, values in domain_stats.items()
        },
        "per_difficulty": {
            key: {"total": len(values), "count": sum(values), "mean": mean(values) if values else 0}
            for key, values in difficulty_stats.items()
        },
    }

    if wandb_project:
        wandb = importlib.import_module("wandb")
        run = wandb.init(project=wandb_project, name=model_name)
        run.log(
            {
                "correct_mean": stats["total"]["correct_mean"],
                "code_render": stats["code"]["correct_code_render"],
                "code_stat_f1": stats["code"]["correct_code_stat_f1"],
                "code_stat_score": stats["code"]["correct_code_stat_score"],
            }
        )
        run.finish()

    return stats


def compare_node_stats(gold_stats: dict, pred_stats: dict) -> tuple[float, float]:
    gold_stats = {key: value for key, value in gold_stats.items() if value > 0}
    pred_stats = {key: value for key, value in pred_stats.items() if value > 0}

    all_components = set(gold_stats.keys()) | set(pred_stats.keys())
    total_score = 0.0
    tp = fp = fn = 0

    for comp in all_components:
        gold_count = gold_stats.get(comp, 0)
        pred_count = pred_stats.get(comp, 0)
        score = 0.0 if gold_count == 0 or pred_count == 0 else min(gold_count, pred_count) / max(gold_count, pred_count)

        total_score += score
        tp += min(gold_count, pred_count)
        fp += max(pred_count - gold_count, 0)
        fn += max(gold_count - pred_count, 0)

    if not all_components:
        return 1.0, 1.0

    overall_score = total_score / len(all_components)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return overall_score, f1


def eval_render_stats(data_rendered: list[dict], gold_statistics: dict) -> list[dict]:
    data_stats = []
    for sample in data_rendered:
        score = 0.0
        f1 = 0.0
        gold = gold_statistics.get(sample["id"])
        pred_stats = sample.get("render", {}).get("statistics")
        if gold and pred_stats:
            score, f1 = compare_node_stats(gold["node_types"], pred_stats["node_types"])
        sample["result"]["correct_stat_score"] = score
        sample["result"]["correct_stat_f1"] = f1
        sample["result"]["correct_render"] = sample["result"]["correct"]
        sample["result"]["correct"] = (sample["result"]["correct"] + f1) / 2
        data_stats.append(sample)

    return data_stats


def build_sample(sample_process: dict) -> tuple[dict, str | None]:
    sample = sample_process["sample"]
    save_dir = sample_process["save_dir"]
    category_key = sample["category_key"]
    lang_key = sample["lang_key"]
    sample_id = sample["id"]
    output_base_path = f"{save_dir}/images/{category_key}_{lang_key}/{sample_id}"
    code = extract_code_answer(sample)

    exception = None
    sample["render"] = {}
    try:
        Renderer = get_renderer_class()
        renderer = Renderer.from_dict(
            renderer=sample_process["renderer"], code=code, output_base_path=output_base_path, category=category_key
        )
        render_result = renderer.render()
        print(f"{output_base_path} - {'success' if render_result.success else 'failed'}")
        sample["render"]["path_code"] = render_result.path_code
        sample["render"]["path_img_1"] = render_result.path_image
        sample["render"]["path_log"] = render_result.path_log
        sample["render"]["tool_1"] = render_result.tool
        sample["render"]["debug_message"] = render_result.debug_message
        sample["render"]["size"] = render_result.size.model_dump()
        sample["render"]["statistics"] = render_result.statistics.model_dump()
        sample["render"]["render_success"] = render_result.success
        sample["result"]["correct"] = render_result.success
        del renderer
    except Exception as error:
        exception = f"\nAn exception occurred for {sample_process['renderer']}: {error}"
        print(exception)
        path_code = f"{save_dir}/code/{category_key}_{lang_key}/{sample_id}.txt"
        save_text(filename=path_code, data=code)
        sample["render"]["path_code"] = path_code
        sample["render"]["render_success"] = False
        sample["result"]["correct"] = False

    return sample, exception


def prepare_output_dirs(save_dir: str, categories: dict) -> None:
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/code", exist_ok=True)
    os.makedirs(f"{save_dir}/images", exist_ok=True)

    for category, content in categories.items():
        for lang_key in content["language"]:
            os.makedirs(f"{save_dir}/images/{category}_{lang_key}", exist_ok=True)
            os.makedirs(f"{save_dir}/code/{category}_{lang_key}", exist_ok=True)


def cleanup_empty_dirs(base_dir: str, subdir: str) -> None:
    for dirpath, dirnames, filenames in os.walk(f"{base_dir}/{subdir}", topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
            except OSError as error:
                print(f"Failed to delete {dirpath}: {error}")


def evaluate_file(
    *,
    input_file: str,
    output_dir: str,
    categories: dict,
    gold_statistics: dict,
    max_workers: int,
    wandb_project: str | None,
) -> None:
    start = time.time()
    model_name = Path(input_file).stem
    save_dir = str(Path(output_dir) / model_name)
    prepare_output_dirs(save_dir, categories)

    data = load_json(filename=input_file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of predictions in {input_file}, got {type(data).__name__}")

    samples_code = []
    samples_other = []
    for raw_sample in data:
        sample = dict(raw_sample)
        sample["result"] = dict(raw_sample.get("result", {}))
        if sample["type"] == "code":
            extract_code_answer(sample)
            samples_code.append(
                {
                    "sample": sample,
                    "save_dir": save_dir,
                    "renderer": categories[sample["category_key"]]["language"][sample["lang_key"]]["renderer"],
                }
            )
        else:
            recompute_non_code_correct(sample)
            samples_other.append(sample)

    dataset_code = []
    dataset_jsonl = f"{save_dir}/dataset.jsonl"
    exceptions_jsonl = f"{save_dir}/exceptions.jsonl"
    check_dirs(dataset_jsonl)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(build_sample, sample_process): sample_process for sample_process in samples_code}

        with open(dataset_jsonl, "w", encoding="utf-8") as dataset_file, open(exceptions_jsonl, "w", encoding="utf-8") as exc_file:
            with create_progress(total=len(futures), desc=f"Rendering {model_name}") as progress:
                for future in as_completed(futures):
                    progress.update(1)
                    result, exc = future.result()
                    dataset_code.append(result)
                    dataset_file.write(f"{json.dumps(result, ensure_ascii=False)}\n")
                    if exc:
                        exc_file.write(f"{exc}\n-------------------------------------\n\n")

    dataset_code = eval_render_stats(dataset_code, gold_statistics)
    dataset = dataset_code + samples_other
    save_json(filename=f"{save_dir}/dataset.json", data=dataset)
    save_json(filename=f"{save_dir}/result.json", data=compute_statistics(dataset, model_name, wandb_project))

    cleanup_empty_dirs(save_dir, "code")
    cleanup_empty_dirs(save_dir, "images")
    print(f"Elapsed time for {model_name}: {time.time() - start}")


def main() -> int:
    args = parse_args()
    categories = load_json(filename=args.categories)
    gold_statistics = load_json(filename=args.gold_statistics)

    for input_file in args.input_files:
        evaluate_file(
            input_file=input_file,
            output_dir=args.output_dir,
            categories=categories,
            gold_statistics=gold_statistics,
            max_workers=args.max_workers,
            wandb_project=args.wandb_project,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
