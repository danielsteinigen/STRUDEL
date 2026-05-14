from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from statistics import mean

from strudel.evaluation.vlm_runtime import (
    DEFAULT_SYSTEM_PROMPT,
    build_request,
    ensure_parent_dir,
    filter_none,
    load_eval_config,
    load_tokenizer,
    needs_tokenizer,
    resolve_generation_api,
)
from strudel.util import check_reasoning, extract_part, save_json

map_answer = {
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
    "United States of America": "America"
}


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the VLM evaluation config YAML.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path prefix for prediction and result JSON files.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Override the dataset path defined in the config.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Override the dataset config name defined in the config.")
    parser.add_argument("--split", type=str, default=None, help="Override the dataset split defined in the config.")
    parser.add_argument("--model-name-or-path", type=str, default=None, help="Override the model name defined in the config.")
    parser.add_argument("--tp-size", type=int, default=None, help="Override the tensor parallel size defined in the config.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on evaluated samples.")
    parser.add_argument("--start-index", type=int, default=None, help="Optional dataset start index.")
    parser.add_argument("--end-index", type=int, default=None, help="Optional dataset end index.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override the system prompt defined in the config.",
    )
    return parser.parse_args()


def compute_statistics(data):
    correct_flags = [sample["result"]["correct"] for sample in data]
    total_correct = sum(correct_flags)
    mean_correct = mean(correct_flags) if correct_flags else 0

    # Per difficulty
    difficulty_stats = defaultdict(list)
    for sample in data:
        difficulty_stats[sample["difficulty"]].append(sample["result"]["correct"])

    difficulty_results = {
        diff: {
            "total": len(vals),
            "count": sum(vals),
            "mean": mean(vals) if vals else 0
        }
        for diff, vals in difficulty_stats.items()
    }

    # Per category
    cat_stats = defaultdict(list)
    for sample in data:
        cat_stats[sample["category_key"]].append(sample["result"]["correct"])

    cat_results = {
        subj: {
            "total": len(vals),
            "count": sum(vals),
            "mean": mean(vals) if vals else 0
        }
        for subj, vals in cat_stats.items()
    }

    # Per domain
    domain_stats = defaultdict(list)
    for sample in data:
        domain = sample["domain"] if sample["domain"] else "business" if sample["category_key"] in ["table", "chart"] else "None"
        domain_stats[domain].append(sample["result"]["correct"])

    domain_results = {
        subj: {
            "total": len(vals),
            "count": sum(vals),
            "mean": mean(vals) if vals else 0
        }
        for subj, vals in domain_stats.items()
    }

    # Per type
    type_stats = defaultdict(list)
    for sample in data:
        type_stats[sample["type"]].append(sample["result"]["correct"])

    type_results = {
        ty: {
            "total": len(vals),
            "count": sum(vals),
            "mean": mean(vals) if vals else 0
        }
        for ty, vals in type_stats.items()
    }

    # Per task type
    task_type_stats = defaultdict(list)
    for sample in data:
        task_type_stats[sample["task_type"]].append(sample["result"]["correct"])

    task_type_results = {
        ty: {
            "total": len(vals),
            "count": sum(vals),
            "mean": mean(vals) if vals else 0
        }
        for ty, vals in task_type_stats.items()
    }

    # Final statistics dictionary
    stats = {
        "total": {
            "samples": len(data),
            "correct_count": total_correct,
            "correct_mean": mean_correct
        },
        "task_type": task_type_results,
        "per_type": type_results,
        "per_category": cat_results,
        "per_domain": domain_results,
        "per_difficulty": difficulty_results
    }

    return stats


def _load_dataset(args, config: dict):
    from datasets import load_dataset

    dataset_cfg = config.get("dataset", {})
    dataset_path = args.dataset or dataset_cfg.get("path", "danielsteinigen/STRUDEL")
    dataset_name = args.dataset_name if args.dataset_name is not None else dataset_cfg.get("name", dataset_cfg.get("config_name"))
    split = args.split or dataset_cfg.get("split", "test")

    if dataset_name:
        dataset = load_dataset(dataset_path, dataset_name, split=split)
    else:
        dataset = load_dataset(dataset_path, split=split)

    if args.start_index is not None and args.end_index is not None and args.end_index > args.start_index:
        dataset = dataset.skip(args.start_index).take(args.end_index - args.start_index)
    elif args.start_index is not None:
        dataset = dataset.skip(args.start_index)

    return dataset


def main() -> int:
    args = get_args()
    config = load_eval_config(args.config)
    dataset = _load_dataset(args, config)
    save_filepath = args.output

    engine_config = filter_none(config["engine"])
    if args.model_name_or_path:
        engine_config["model"] = args.model_name_or_path
    if args.tp_size is not None:
        engine_config["tensor_parallel_size"] = args.tp_size

    sampling_config = filter_none(config["sampling"])
    model_name = engine_config["model"]
    generation = config["generation"]
    generation_api = resolve_generation_api(config)
    system_prompt = args.system_prompt or config.get("prompts", {}).get("system", DEFAULT_SYSTEM_PROMPT)
    user_field = generation.get("user_field", "user_prompt")
    image_field = generation.get("image_field", "image")

    ensure_parent_dir(save_filepath)

    print("Loading Model ...")
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(**sampling_config)
    llm = LLM(**engine_config)
    tokenizer = None
    if needs_tokenizer(config):
        tokenizer = load_tokenizer(model_name, trust_remote_code=engine_config.get("trust_remote_code", False))

    prompts = []
    outputs = []

    print("Building prompts ...")
    for sample in dataset:
        prompts.append(
            build_request(
                config=config,
                system_prompt=system_prompt,
                user_prompt=sample[user_field],
                image=sample[image_field],
                tokenizer=tokenizer,
            )
        )
        if args.max_samples is not None and len(prompts) >= args.max_samples:
            break

    print("Run generation ...")
    print(f"LEN prompts: {len(prompts)}")
    if generation_api == "generate":
        outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    else:
        outputs = llm.chat(messages=prompts, sampling_params=sampling_params)
    print(f"LEN outputs: {len(outputs)}")

    print("Processing output ...")
    cnt_not_stop = 0
    dataset_final = []
    for data, out in zip(dataset, outputs):
        if args.max_samples is not None and len(dataset_final) >= args.max_samples:
            break

        sample = dict(data)
        correct = False
        out_raw = out.outputs[0].text.strip()
        out_no_reas = check_reasoning(out_raw)
        answer_gold = data["answer"].strip()
        if data["category_key"] == "geo" and answer_gold in map_answer:
            answer_gold = map_answer[answer_gold]
        if sample["type"] == "code":
            answers = extract_part(out_no_reas, "```", "```", False, True).strip()
            correct = True
        else:
            # answer = out_no_reas.split("\n")[0].split(")")[0].split(":")[0]
            out_no_reas = out_no_reas.replace("*","").replace("<|begin_of_box|>","").replace("<|end_of_box|>","").replace("boxed","")
            num_chars = len(answer_gold)+15
            answers = [out_no_reas[:num_chars]]
            if num_chars < len(out_no_reas):
                answers.append(out_no_reas[len(out_no_reas)-num_chars:])
                if "nswer:" in out_no_reas:
                    answers.append(out_no_reas.split("nswer:")[1][:num_chars])
                if "answer is" in out_no_reas:
                    answers.append(out_no_reas.split("answer is")[1][:num_chars])

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
                    check_gold = check_gold.replace("yes","true").replace("no","false")
                    if check_gold in check_pred:
                        correct = True
                        break

        sample["result"] = {
            "prompt": out.prompt,
            "generation": out_raw,
            "generation_split": out_no_reas,
            "stop": out.outputs[0].finish_reason,
            "answer": answers,
            "correct": correct,
        }
        if image_field in sample:
            del sample[image_field]
        dataset_final.append(sample)
        if out.outputs[0].finish_reason != "stop": cnt_not_stop += 1

    print(f"Finish reason not stop: {cnt_not_stop}.")
    results = compute_statistics(dataset_final)
    results["cnt_not_stop"] = cnt_not_stop
    save_json(filename=f"{save_filepath}.json", data=dataset_final)
    save_json(filename=f"{save_filepath}_result.json", data=results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
