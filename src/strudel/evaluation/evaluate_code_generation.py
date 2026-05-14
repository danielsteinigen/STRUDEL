import argparse
import csv
import os
from collections import defaultdict
from statistics import mean

import wandb
from tqdm import tqdm

from strudel.util import load_jsonl, load_text

languages_cnt_chars = ["logic_bool", "logic_symb", "newick", "fasta", "vienna", "smiles", "smarts", "smarts_react", "fen", "abc"]


def count_lines_in_file(filepath):
    if not filepath or not os.path.exists(filepath):
        print(f"Error filepath: {filepath}")
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_chars_in_file(filepath):
    if not filepath or not os.path.exists(filepath):
        print(f"Error filepath: {filepath}")
        return 0
    return len(load_text(filename=filepath).strip())


def count_words(text):
    if not text:
        return 0
    return len(str(text.strip()).split())


def is_significantly_greater(new, base, margin=0.1):
    return new > base * (1 + margin)


def format_row(row):
    return [f"{cell:.4f}".replace(".", ",") if isinstance(cell, float) else cell for cell in row]


def main():
    parser = argparse.ArgumentParser(description="Evaluate code-generation outputs grouped by FRL.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Directories containing evaluation dataset.jsonl files")
    args = parser.parse_args()

    save_dirs = args.input_dirs

    for filepath in save_dirs:
        dataset = load_jsonl(f"{filepath}/dataset.jsonl")

        save_filepath = f"{filepath}/result"
        model_name = filepath.split("/")[-1].split(".")[0]
        print(f"Evaluating {len(dataset)} samples from {model_name}.")
        approach = f"end2end_{filepath.split('-')[-1]}"
        wandb.init(project="code-eval", name=model_name, config={"model": model_name, "approach": approach})

        # --- Part 1: Per approach statistics ---
        approach_stats = defaultdict(
            lambda: {
                "desc_words": [],
                "prob_words": [],
                "code_lines": [],
                "no_code": 0,
                "image_count": [],
                "image_count_low": [],
                "image_count_medium": [],
                "image_count_high": [],
                "compl_desc_score": 0.0,
                "compl_code_score": 0.0,
                "compl_overall_score": 0.0,
            }
        )

        # --- Part 2: Per approach + complexity + language ---
        group_stats = defaultdict(lambda: {"desc_words": [], "code_lines": [], "image_count": []})

        image_counts_by_language = defaultdict(lambda: defaultdict(list))

        groups_complex = {
            approach: defaultdict(dict),
        }

        approach_stats[approach]["no_code"] = 9600 - len(dataset)

        for obj in tqdm(dataset, total=len(dataset)):
            inp = obj.get("input")
            id_compl = inp.get("id_compl")
            path_img = obj.get("path_img_1")
            complexity = inp.get("complexity", "").strip()
            language = inp.get("lang_key", "").strip()
            category = inp.get("category_key", "").strip()

            if complexity not in ["low", "medium", "high"]:
                print(f"Invalid complexity: {complexity}")
                continue
            if language == "hmm":
                continue

            # --- Collect stats ---
            desc_word_count = count_words(obj.get("description", ""))
            prob_word_count = count_words(obj.get("problem", ""))
            if language in languages_cnt_chars:
                code_lines_count = count_chars_in_file(obj.get("path_code", ""))
            else:
                code_lines_count = count_lines_in_file(obj.get("path_code", ""))

            # Update per (approach, complexity, category, language)
            key = (approach, complexity, category, language)
            # Update per-approach
            if desc_word_count > 0:
                approach_stats[approach]["desc_words"].append(desc_word_count)
                group_stats[key]["desc_words"].append(desc_word_count)

            if prob_word_count > 0:
                approach_stats[approach]["prob_words"].append(prob_word_count)

            if code_lines_count > 0:
                approach_stats[approach]["code_lines"].append(code_lines_count)
                group_stats[key]["code_lines"].append(code_lines_count)

            if id_compl:
                groups_complex[approach][id_compl][complexity] = (desc_word_count, code_lines_count)

            # Check for non-empty path_img_1
            if path_img and str(path_img).strip():
                group_stats[key]["image_count"].append(1)
                image_counts_by_language[f"{category}_{language}"][approach].append(1)
                approach_stats[approach]["image_count"].append(1)
                approach_stats[approach][f"image_count_{complexity}"].append(1)
            else:
                if code_lines_count > 0:
                    image_counts_by_language[f"{category}_{language}"][approach].append(0)
                    group_stats[key]["image_count"].append(0)
                    approach_stats[approach]["image_count"].append(0)
                    approach_stats[approach][f"image_count_{complexity}"].append(0)
                else:
                    approach_stats[approach]["no_code"] += 1

        # Evaluate complexity comparisons
        for appr, groups_appr in groups_complex.items():
            compl_desc_results = []
            compl_code_results = []
            for idc, group in groups_appr.items():
                if all(k in group for k in ("low", "medium", "high")):
                    desc_low, code_low = group["low"]
                    desc_med, code_med = group["medium"]
                    desc_high, code_high = group["high"]

                    # Description word comparisons
                    if desc_low > 0 and desc_med > 0 and desc_high > 0:
                        compl_desc_results.append(1 if is_significantly_greater(desc_med, desc_low) else 0)
                        compl_desc_results.append(1 if is_significantly_greater(desc_high, desc_med) else 0)
                        compl_desc_results.append(1 if is_significantly_greater(desc_high, desc_low) else 0)

                    # Code line comparisons
                    if code_low > 0 and code_med > 0 and code_high > 0:
                        compl_code_results.append(1 if is_significantly_greater(code_med, code_low) else 0)
                        compl_code_results.append(1 if is_significantly_greater(code_high, code_med) else 0)
                        compl_code_results.append(1 if is_significantly_greater(code_high, code_low) else 0)

            approach_stats[appr]["compl_desc_score"] = mean(compl_desc_results) if compl_desc_results else 0
            approach_stats[appr]["compl_code_score"] = mean(compl_code_results) if compl_code_results else 0
            approach_stats[appr]["compl_overall_score"] = (
                mean(compl_desc_results + compl_code_results) if (compl_desc_results + compl_code_results) else 0
            )

        # --- Transposed Per-approach CSV ---
        overall_metrics = {}
        with open(f"{save_filepath}_approach.csv", "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["metric"] + [approach])
            metrics = ["desc_words", "prob_words", "code_lines"]
            for metric in metrics:
                row = [f"mean_{metric}"]
                values = approach_stats[approach]
                stat = mean(values[metric]) if values[metric] else 0
                row_val = int(stat)
                row.append(row_val)
                overall_metrics[f"mean_{metric}"] = row_val
                writer.writerow(format_row(row))

            for compl in ["compl_desc_score", "compl_code_score", "compl_overall_score"]:
                row = [f"{compl}"]
                values = approach_stats[approach]
                row_val = round(values[f"{compl}"], 4)
                row.append(row_val)
                overall_metrics[compl] = row_val
                writer.writerow(format_row(row))

            row = ["No_Code"]
            values = approach_stats[approach]
            row_val = values["no_code"]
            row.append(row_val)
            overall_metrics["No_Code"] = row_val
            writer.writerow(format_row(row))

            row = ["Image_Total"]
            values = approach_stats[approach]
            row_val = len(values["image_count"])
            row.append(row_val)
            overall_metrics["Image_Total"] = row_val
            writer.writerow(format_row(row))

            row = ["Image_Count"]
            values = approach_stats[approach]
            row_val = sum(values["image_count"])
            row.append(row_val)
            overall_metrics["Image_Count"] = row_val
            writer.writerow(format_row(row))

            row = ["Image_Mean"]
            values = approach_stats[approach]
            row_val = round(mean(values["image_count"]), 4) if values["image_count"] else 0
            row.append(row_val)
            overall_metrics["Image_Mean"] = row_val
            writer.writerow(format_row(row))

            row = ["Image_Mean_No_Code"]
            values = approach_stats[approach]
            row_val = (
                round(sum(values["image_count"]) / (len(values["image_count"]) + values["no_code"]), 4) if values["image_count"] else 0
            )
            row.append(row_val)
            overall_metrics["Image_Mean_No_Code"] = row_val
            writer.writerow(format_row(row))

            for compl in ["low", "medium", "high"]:
                row = [f"Image_Mean_{compl}"]
                values = approach_stats[approach]
                row_val = round(mean(values[f"image_count_{compl}"]), 4) if values[f"image_count_{compl}"] else 0
                row.append(row_val)
                overall_metrics[f"Image_Mean_{compl}"] = row_val
                writer.writerow(format_row(row))
        wandb.log(overall_metrics)

        # --- Per-group CSV ---
        group_metrics = {}
        with open(f"{save_filepath}_complexity.csv", "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                ["approach", "complexity", "language", "mean_desc_words", "mean_code_lines", "image_total", "image_count", "image_mean"]
            )
            for (approach, complexity, category, language), stats in group_stats.items():
                mean_desc_words = (int(mean(stats["desc_words"])) if stats["desc_words"] else 0,)
                mean_code_lines = (int(mean(stats["code_lines"])) if stats["code_lines"] else 0,)
                image_total = (len(stats["image_count"]),)
                image_count = (sum(stats["image_count"]),)
                image_mean = round(mean(stats["image_count"]), 4) if stats["image_count"] else 0
                writer.writerow(
                    format_row(
                        [
                            approach,
                            complexity,
                            f"{category}_{language}",
                            mean_desc_words,
                            mean_code_lines,
                            image_total,
                            image_count,
                            image_mean,
                        ]
                    )
                )
                group_metrics[f"{category}_{language}/{complexity}/mean_desc_words"] = mean_desc_words[0]
                group_metrics[f"{category}_{language}/{complexity}/mean_code_lines"] = mean_code_lines[0]
                group_metrics[f"{category}_{language}/{complexity}/image_total"] = image_total[0]
                group_metrics[f"{category}_{language}/{complexity}/image_count"] = image_count[0]
                group_metrics[f"{category}_{language}/{complexity}/image_mean"] = image_mean
        wandb.log(group_metrics)

        # --- New: Image count per language and approach ---
        with open(f"{save_filepath}_count.csv", "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["language"] + [approach])
            for lang, approach_counts in image_counts_by_language.items():
                row = [lang] + [round(mean(approach_counts.get(approach, [0])), 4)]
                writer.writerow(format_row(row))
                wandb.log({f"{lang}/image_mean": round(mean(approach_counts.get(approach, [0])), 4)})

        print("CSV files written: mean_stats_per_approach.csv, mean_stats_by_group.csv, image_counts_by_language.csv")

        wandb.finish()


if __name__ == "__main__":
    main()
