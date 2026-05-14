from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandSpec:
    script_path: str
    description: str
    usage: str


REPO_ROOT = Path(__file__).resolve().parents[2]

COMMANDS: dict[tuple[str, str], CommandSpec] = {
    ("personas", "generate"): CommandSpec(
        script_path="src/strudel/data_generator/persona_data_generator.py",
        description="Generate raw persona candidates from FineWebEdu.",
        usage="strudel personas generate --config CONFIG --output OUTPUT [--max-samples N] [--data-batch-size N] [--start-index N]",
    ),
    ("personas", "query"): CommandSpec(
        script_path="src/strudel/data_generator/persona_query_data_generator.py",
        description="Generate semantic search queries for each visualization category.",
        usage="strudel personas query --config CONFIG --input CATEGORIES_JSON --output OUTPUT [--max-samples N] [--data-batch-size N]",
    ),
    ("personas", "filter"): CommandSpec(
        script_path="src/strudel/personas/filter_personas_semantic_hf.py",
        description="Filter persona candidates into category-specific subsets.",
        usage="strudel personas filter --input INPUT_JSONL --output OUTPUT_DIR [--query-path PATH] [--index-type TYPE] [--embedding-model MODEL]",
    ),
    ("dataset", "generate"): CommandSpec(
        script_path="src/strudel/data_generator/structured_data_generator.py",
        description="Generate STRUDEL samples from personas.",
        usage="strudel dataset generate --config CONFIG --input PERSONAS_JSONL --output OUTPUT [--max-samples N] [--data-batch-size N] [--start-index N] [--end-index N] [--use-harmony]",
    ),
    ("dataset", "render"): CommandSpec(
        script_path="src/strudel/render_proxy.py",
        description="Invoke Structivize rendering for generated samples.",
        usage="strudel dataset render --render-script /path/to/structivize/src/structivize/render_batch.py [render args...]",
    ),
    ("dataset", "filter"): CommandSpec(
        script_path="src/strudel/filtering/filter_generations.py",
        description="Filter rendered generations into a curated dataset.",
        usage="strudel dataset filter --input-dirs DIR [DIR ...] --output-dir OUTPUT_DIR [--categories PATH] [--copy-images] [--max-dups-code N] [--max-dups-img N]",
    ),
    ("dataset", "score"): CommandSpec(
        script_path="src/strudel/data_generator/scoring_data_generator.py",
        description="Score generated samples with LLM-as-a-Judge.",
        usage="strudel dataset score --config CONFIG --input DATASET_JSONL --output OUTPUT [--max-samples N] [--data-batch-size N] [--start-index N]",
    ),
    ("dataset", "split"): CommandSpec(
        script_path="src/strudel/filtering/split_dataset.py",
        description="Split curated datasets into refinement subsets.",
        usage="strudel dataset split --qa-input-dirs DIR [DIR ...] --ps-input-dirs DIR [DIR ...] --output-dir OUTPUT_DIR [--categories PATH]",
    ),
    ("dataset", "qa"): CommandSpec(
        script_path="src/strudel/data_generator/qa_data_generator.py",
        description="Generate question-answer refinement data.",
        usage="strudel dataset qa --config CONFIG --input DATASET_JSONL --output OUTPUT [--max-samples N] [--data-batch-size N] [--start-index N]",
    ),
    ("dataset", "caption"): CommandSpec(
        script_path="src/strudel/data_generator/caption_data_generator.py",
        description="Generate caption refinement data.",
        usage="strudel dataset caption --config CONFIG --input DATASET_JSONL --output OUTPUT [--max-samples N] [--data-batch-size N] [--start-index N]",
    ),
    ("dataset", "assemble"): CommandSpec(
        script_path="src/strudel/filtering/assemble_dataset.py",
        description="Assemble the final training dataset from refinement subsets.",
        usage="strudel dataset assemble --input-dir INPUT_DIR --output-dir OUTPUT_DIR [--categories PATH]",
    ),
    ("eval", "codegen"): CommandSpec(
        script_path="src/strudel/evaluation/evaluate_code_generation.py",
        description="Evaluate code-generation outputs grouped by FRL.",
        usage="strudel eval codegen --input-dirs DIR [DIR ...]",
    ),
    ("eval", "codetrans"): CommandSpec(
        script_path="src/strudel/evaluation/evaluate_code_translation.py",
        description="Render and score code-translation outputs from STRUDEL evaluation runs.",
        usage="strudel eval codetrans --input-files FILE [FILE ...] --output-dir DIR --gold-statistics PATH [--categories PATH] [--max-workers N] [--wandb-project NAME]",
    ),
    ("eval", "strudel"): CommandSpec(
        script_path="src/strudel/evaluation/evaluate_strudel.py",
        description="Evaluate a VLM on the STRUDEL benchmark using a VLM config.",
        usage="strudel eval strudel --config CONFIG --output OUTPUT [--dataset PATH] [--dataset-name NAME] [--split SPLIT] [--model-name-or-path MODEL] [--tp-size N] [--max-samples N] [--start-index N] [--end-index N]",
    ),
    ("eval", "testset"): CommandSpec(
        script_path="src/strudel/evaluation/evaluate_testset.py",
        description="Evaluate a model on the STRUDEL test set.",
        usage="strudel eval testset --model_name_or_path MODEL --output_path OUTPUT [--is_reasoning_model] [--use_think_tag]",
    ),
}

GROUP_ORDER = ["personas", "dataset", "train", "eval"]


def _print_help(group: str | None = None) -> int:
    if group is None:
        print("Usage: strudel <group> <command> [args...]")
        print()
        print("Available command groups:")
        for group_name in GROUP_ORDER:
            print(f"  {group_name}")
        print()
        print("Run 'strudel <group> --help' to list commands in a group.")
        return 0

    group_commands = [(command, spec) for (group_name, command), spec in COMMANDS.items() if group_name == group]
    if not group_commands:
        print(f"Unknown command group: {group}", file=sys.stderr)
        return 1

    print(f"Usage: strudel {group} <command> [args...]")
    print()
    print("Available commands:")
    for command_name, spec in group_commands:
        print(f"  {command_name:<10} {spec.description}")
    return 0


def _print_command_help(group: str, command: str, spec: CommandSpec) -> int:
    print(f"Usage: {spec.usage}")
    print()
    print(spec.description)
    print()
    print(f"Implementation target: {spec.script_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help"}:
        return _print_help()

    if len(args) == 1 and args[0] not in {group for group, _ in COMMANDS}:
        print(f"Unknown command group: {args[0]}", file=sys.stderr)
        return _print_help()

    if len(args) == 1 or args[1] in {"-h", "--help"}:
        return _print_help(args[0])

    key = (args[0], args[1])
    spec = COMMANDS.get(key)
    if spec is None:
        print(f"Unknown command: {args[0]} {args[1]}", file=sys.stderr)
        return _print_help(args[0])

    if len(args) == 2 or args[2] in {"-h", "--help"}:
        return _print_command_help(args[0], args[1], spec)

    script_path = REPO_ROOT / spec.script_path
    if not script_path.exists():
        print(f"Command target not found: {script_path}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    python_paths = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    command = [sys.executable, str(script_path), *args[2:]]
    if script_path.suffix == ".sh":
        command = ["bash", str(script_path), *args[2:]]

    result = subprocess.run(command, cwd=REPO_ROOT, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
