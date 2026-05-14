import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import zip_longest

from datasets import concatenate_datasets, load_dataset

from strudel.filtering.mappings import get_map_key, map_domain, question_templates
from strudel.prompt_templates.templates_refinement import (
    ASSISTANT_ASSOCIATION,
    ASSISTANT_ASSOCIATION_NONE,
    ASSISTANT_CODE,
    ASSISTANT_CONSISTENCY,
    ASSISTANT_STRUCTURAL,
    PROMPT_ASSOCIATION_SHORT,
    PROMPT_CONSISTENCY_SHORT,
    USER_ASSOCIATION_CAPTION,
    USER_ASSOCIATION_PERSONA,
    USER_CAPTION,
    USER_CODE,
    USER_CONSISTENCY_PROBLEM,
    USER_CONSISTENCY_SOLUTION,
    USER_DESCRIPTION,
)
from strudel.util import load_json, save_json

random.seed(42)
save_filepath = f"/data/dataset_v10/dataset_assembled_new"
complexities = ["low", "medium", "high"]
idx2aw = {0: "A", 1: "B", 2: "C", 3: "D"}

format_structural = "\nAnswer the question using a single integer number."
format_association = "\nAnswer with the option's letter from the given choices directly."
format_consistency = "\nAnswer the question with a single word."


def save_stats_train(ds, path):
    print("Save statistics ...")
    type_counts = Counter(ds["type"])
    diff_counts = Counter(ds["difficulty"])
    domain_counts = Counter(ds["domain"])
    cat_counts = Counter(ds["category_key"])
    stats = {
        "total_count": len(ds),
        "type_counts": dict(type_counts),
        "difficulty_counts": dict(diff_counts),
        "domain_counts": dict(domain_counts),
        "category_counts": dict(cat_counts),
    }
    save_json(path, stats)


def generate_structural_question(batch):
    convs = []
    for statistics_item, category_item, lang_item in zip(batch["statistics"], batch["category_key"], batch["lang_key"]):
        conv = []
        idx = 0
        node_types = json.loads(statistics_item)["node_types"]
        map_key = get_map_key(category_item, lang_item)
        map_key = (
            map_key
            if map_key not in ["class", "sequence", "activity", "component", "usecase", "state", "requirement", "bpmn", "plotly"]
            else "uml"
        )
        if node_types and map_key in question_templates:
            if lang_item == "nn_onnx_graph" and "op_types" in node_types:
                node_types = node_types["op_types"]
            for component, cnt in node_types.items():
                if cnt > 0 and component not in [
                    "activations",
                    "deactivations",
                    "labeled_transitions",
                    "end_transitions",
                    "start_transitions",
                    "start_events",
                    "end_events",
                    "conditional_flows",
                ]:
                    rand_max = 9 if map_key != "uml" else 31
                    question = question_templates[map_key if not (component == "sequences" and lang_item == "fasta") else "fasta_seq"][
                        random.randint(0, rand_max)
                    ]
                    if category_item == "chart":
                        question = question.replace("diagram", "chart")
                    if category_item == "table":
                        question = question.replace("diagram", "table")

                    assistant_msg = ASSISTANT_STRUCTURAL[random.randint(0, 9)].format(component=component, cnt=str(cnt))
                    resp_format = ""
                    if random.randint(0, 2) == 0:
                        resp_format = format_structural
                        assistant_msg = str(cnt)
                    conv.append(
                        {
                            "id": idx,
                            "user": f"{question.format(component=component)}{resp_format}",
                            "assistant": assistant_msg,
                            "subject": component,
                            "gold_answer": cnt,
                        }
                    )
                    idx += 1
        convs.append(conv)

    return {"conversation_structural": convs}


def assemble_association_question(id, category, cat_name, subject, categories, choices, gold_answer, prompt, use_gold, is_difficult):
    cnt = 3 if use_gold else 4
    answer = "NONE"
    answer_idx = -1
    options = [gold_answer]

    if is_difficult:
        categ = random.sample(categories, cnt - 1)
        categ += [category]
    else:
        categ = random.sample(categories, cnt)

    while gold_answer in options:
        options = [random.choice(choices[cat])[subject] for cat in categ]
    random.shuffle(options)

    if use_gold:
        answer_idx = random.randint(0, 3)
        options.insert(answer_idx, gold_answer)
        answer = idx2aw[answer_idx]

    assistant_msg = (
        ASSISTANT_ASSOCIATION[random.randint(0, 9)].format(answer=answer, subject=subject, category=cat_name)
        if answer_idx > -1
        else ASSISTANT_ASSOCIATION_NONE[random.randint(0, 9)].format(subject=subject, category=cat_name)
    )
    resp_format = ""
    if random.randint(0, 2) == 0:
        resp_format = format_association
        assistant_msg = answer if answer_idx > -1 else "None"
    return {
        "id": f"{id}",
        "user": PROMPT_ASSOCIATION_SHORT.format(
            question=prompt, option_a=options[0], option_b=options[1], option_c=options[2], option_d=options[3], resp_format=resp_format
        ),
        "assistant": assistant_msg,
        "difficulty": "low" if not is_difficult else ("medium" if use_gold else "high"),
        "subject": subject,
        "options": options,
        "answer_idx": answer_idx,
        "gold_answer": answer,
    }


def generate_association_question(batch, categories, use_persona=True):
    category_dict = defaultdict(list)

    convs = []
    if use_persona:
        for item in batch["input"]:
            category_dict[item["category_key"]].append({"persona": item["persona"]["persona"]})
        category_dict = dict(category_dict)
        for category_item, cat_name_item, input_item in zip(batch["category_key"], batch["category_name"], batch["input"]):
            conv = []
            cont_categ_pers = [
                cont_cat for cont_cat in categories[category_item]["benchmark"]["contrary_personas"] if cont_cat in category_dict
            ]
            if input_item["domain"]:
                cont_categ_pers = [
                    cont_cat
                    for cont_cat in cont_categ_pers
                    if input_item["domain"] not in categories[cont_cat]["benchmark"]["exclude_domains"]
                ]

            for idx, (gold, diff) in enumerate(zip([True, False], [False, False])):
                conv.append(
                    assemble_association_question(
                        id=idx,
                        category=category_item,
                        cat_name=cat_name_item[:-1] if cat_name_item.endswith("s") else cat_name_item,
                        subject="persona",
                        categories=cont_categ_pers,
                        choices=category_dict,
                        gold_answer=input_item["persona"]["persona"],
                        prompt=USER_ASSOCIATION_PERSONA[random.randint(0, 9)],
                        use_gold=gold,
                        is_difficult=diff,
                    )
                )
            convs.append(conv)

    else:
        for (
            item_cat,
            item_cap,
        ) in zip(batch["category_key"], batch["caption"]):
            category_dict[item_cat].append({"caption": item_cap})
        category_dict = dict(category_dict)
        for category_item, cat_name_item, caption_item in zip(batch["category_key"], batch["category_name"], batch["caption"]):
            conv = []
            cont_categ = [
                cont_cat for cont_cat in categories[category_item]["benchmark"]["contrary_categories"] if cont_cat in category_dict
            ]
            for idx, (gold, diff) in enumerate(zip([True, False, True, False], [False, False, True, True])):
                conv.append(
                    assemble_association_question(
                        id=idx,
                        category=category_item,
                        cat_name=cat_name_item[:-1] if cat_name_item.endswith("s") else cat_name_item,
                        subject="caption",
                        categories=cont_categ,
                        choices=category_dict,
                        gold_answer=caption_item,
                        prompt=USER_ASSOCIATION_CAPTION[random.randint(0, 9)],
                        use_gold=gold,
                        is_difficult=diff,
                    )
                )
            convs.append(conv)

    return {"conversation_association": convs}


def assemble_consistency_question(id, category, cat_name, sample, subject, categories, choices, prompt, use_gold, is_difficult):
    text = sample
    if not use_gold:
        categ = category
        if not is_difficult:
            categ = random.choice(categories)
        while text == sample:
            text = random.choice(choices[categ])[subject]
    answer = "Yes" if use_gold else "No"

    assistant_msg = ASSISTANT_CONSISTENCY[random.randint(0, 9)].format(
        answer=answer, subject=subject, neg=("" if use_gold else "not "), category=cat_name
    )
    resp_format = ""
    if random.randint(0, 2) == 0:
        resp_format = format_consistency
        assistant_msg = answer
    return {
        "id": f"{id}",
        "user": PROMPT_CONSISTENCY_SHORT.format(question=prompt, text=text, subject=subject.capitalize(), resp_format=resp_format),
        "assistant": assistant_msg,
        "difficulty": "low" if not is_difficult else "high",
        "subject": subject,
        "options": [text],
        "answer_idx": 1 if use_gold else 0,
        "gold_answer": answer,
    }


def generate_consistency_question(batch, categories, description: bool = True):
    category_dict = defaultdict(list)
    for item_cat, item_prob, item_desc in zip_longest(
        batch["category_key"], batch["problem"], batch.get("description", []), fillvalue=None
    ):
        category_dict[item_cat].append({"problem": item_prob, "description": item_desc})
    category_dict = dict(category_dict)

    convs = []
    for category_item, cat_name_item, prob_item, desc_item in zip_longest(
        batch["category_key"], batch["category_name"], batch["problem"], batch.get("description", []), fillvalue=None
    ):
        conv = []
        cont_categ = [cont_cat for cont_cat in categories[category_item]["benchmark"]["contrary_categories"] if cont_cat in category_dict]
        idx = 0
        for gold, diff in zip([True, False, False], [False, False, True]):
            conv.append(
                assemble_consistency_question(
                    id=idx,
                    category=category_item,
                    cat_name=cat_name_item[:-1] if cat_name_item.endswith("s") else cat_name_item,
                    sample=prob_item,
                    subject="problem",
                    categories=cont_categ,
                    choices=category_dict,
                    prompt=USER_CONSISTENCY_PROBLEM[random.randint(0, 9)],
                    use_gold=gold,
                    is_difficult=diff,
                )
            )
            idx += 1
            if description:
                conv.append(
                    assemble_consistency_question(
                        id=idx,
                        category=category_item,
                        cat_name=cat_name_item[:-1] if cat_name_item.endswith("s") else cat_name_item,
                        sample=desc_item,
                        subject="description",
                        categories=cont_categ,
                        choices=category_dict,
                        prompt=USER_CONSISTENCY_SOLUTION[random.randint(0, 9)],
                        use_gold=gold,
                        is_difficult=diff,
                    )
                )
                idx += 1
        convs.append(conv)

    return {"conversation_consistency": convs}


def select_structural_question(sample):
    conv_sample = random.sample(sample["conversation_structural"], 1)[0]
    return {
        "type": "structural",
        "user": conv_sample["user"],
        "assistant": conv_sample["assistant"],
        "difficulty": sample["input"]["complexity"],
        "subject": conv_sample["subject"],
        "gold_answer": conv_sample["gold_answer"],
    }


def select_association_question(sample):
    conv_sample = random.sample(sample["conversation_association"], 1)[0]
    return {
        "type": "association",
        "user": conv_sample["user"],
        "assistant": conv_sample["assistant"],
        "difficulty": conv_sample["difficulty"],
        "subject": conv_sample["subject"],
        "gold_answer": conv_sample["gold_answer"],
        "options": conv_sample["options"],
        "answer_idx": conv_sample["answer_idx"],
    }


def select_consistency_question(sample):
    sample_list = (
        sample["conversation_consistency"] + sample["conversation_consistency"][:2]
    )  # duplicate first two samples to have an even distribution between cosistency yes vs. no
    conv_sample = random.sample(sample_list, 1)[0]
    return {
        "type": "consistency",
        "user": conv_sample["user"],
        "assistant": conv_sample["assistant"],
        "difficulty": conv_sample["difficulty"],
        "subject": conv_sample["subject"],
        "gold_answer": conv_sample["gold_answer"],
        "options": conv_sample["options"],
        "answer_idx": conv_sample["answer_idx"],
    }


def main():
    global save_filepath

    parser = argparse.ArgumentParser(description="Assemble the final dataset.")
    parser.add_argument("--input-dir", required=True, help="Directory containing the intermediate refinement datasets")
    parser.add_argument("--output-dir", required=True, help="Directory where the assembled dataset is written")
    parser.add_argument("--categories", default="diagram_categories.json", help="Path to category definitions JSON")
    args = parser.parse_args()

    input_dir = args.input_dir
    save_filepath = args.output_dir
    categories = load_json(filename=args.categories)

    ds_qa_prob = load_dataset("json", data_files=f"{input_dir}/dataset_qa_problem.jsonl", split="train")
    ds_ps_desc = load_dataset("json", data_files=f"{input_dir}/dataset_ps_description.jsonl", split="train")
    ds_ps_capt = load_dataset("json", data_files=f"{input_dir}/dataset_ps_caption_result_post.jsonl", split="train")
    ds_ass_per = load_dataset("json", data_files=f"{input_dir}/dataset_association_persona.jsonl", split="train")
    ds_ass_cap = load_dataset("json", data_files=f"{input_dir}/dataset_association_caption_result_post.jsonl", split="train")
    ds_consist = load_dataset("json", data_files=f"{input_dir}/dataset_consistency.jsonl", split="train")
    ds_llm_gen = load_dataset("json", data_files=f"{input_dir}/dataset_llm_qa_result_post.jsonl", split="train")
    ds_structu = load_dataset("json", data_files=f"{input_dir}/dataset_structural.jsonl", split="train")
    ds_transla = load_dataset("json", data_files=f"{input_dir}/dataset_code_translate.jsonl", split="train")

    print("Create QA problem questions ...")
    ds_qa_prob = ds_qa_prob.add_column("type", ["qa_problem"] * len(ds_qa_prob))
    ds_qa_prob = ds_qa_prob.map(lambda sample: {"user": sample["problem"]})
    ds_qa_prob = ds_qa_prob.map(lambda sample: {"assistant": sample["answer"]})
    ds_qa_prob = ds_qa_prob.map(lambda sample: {"difficulty": f"{sample['input']['complexity']}"})

    print("Create description questions ...")
    ds_ps_desc = ds_ps_desc.add_column("type", ["description"] * len(ds_ps_desc))
    ds_ps_desc = ds_ps_desc.map(lambda sample: {"user": USER_DESCRIPTION[random.randint(0, 9)]})
    ds_ps_desc = ds_ps_desc.map(lambda sample: {"assistant": sample["description"]})
    ds_ps_desc = ds_ps_desc.map(lambda sample: {"difficulty": f"{sample['input']['complexity']}"})

    print("Create captioning questions ...")
    ds_ps_capt = ds_ps_capt.add_column("type", ["captioning"] * len(ds_ps_capt))
    ds_ps_capt = ds_ps_capt.map(lambda sample: {"user": USER_CAPTION[random.randint(0, 9)]})
    ds_ps_capt = ds_ps_capt.map(lambda sample: {"assistant": sample["caption"]})
    ds_ps_capt = ds_ps_capt.map(lambda sample: {"difficulty": f"{sample['input']['complexity']}"})

    print("Create code questions ...")
    ds_transla = ds_transla.add_column("type", ["code"] * len(ds_transla))
    ds_transla = ds_transla.map(lambda sample: {"user": USER_CODE[random.randint(0, 9)]})
    ds_transla = ds_transla.map(
        lambda sample: {
            "assistant": f"{ASSISTANT_CODE[random.randint(0, 9)].format(language=sample['lang_name'])}\n```\n{sample['code']}\n```"
        }
    )
    ds_transla = ds_transla.map(lambda sample: {"difficulty": f"{sample['input']['complexity']}"})
    ds_transla = ds_transla.shuffle(seed=42).select(range(len(ds_transla) // 2))

    print("Create QA detail questions ...")
    ds_llm_gen = ds_llm_gen.add_column("type", ["qa_detail"] * len(ds_llm_gen))
    ds_llm_gen = ds_llm_gen.map(lambda sample: {"user": sample["llm_user"]})
    ds_llm_gen = ds_llm_gen.map(lambda sample: {"assistant": sample["llm_assistant"]})
    ds_llm_gen = ds_llm_gen.map(lambda sample: {"difficulty": f"{sample['input']['complexity']}"})

    print("Create structural questions ...")
    ds_structu = ds_structu.map(generate_structural_question, batched=True, batch_size=10000, desc="Create structural questions")
    ds_structu_not = ds_structu.filter(lambda sample: not sample["conversation_structural"])
    ds_structu = ds_structu.filter(lambda sample: sample["conversation_structural"])
    ds_structu = ds_structu.map(select_structural_question)
    print(len(ds_structu))
    ds_structu_not.to_json(f"{save_filepath}/no_struct.jsonl")

    print("Create association questions ...")
    ds_ass_per = ds_ass_per.map(
        generate_association_question,
        fn_kwargs={"categories": categories, "use_persona": True},
        batched=True,
        batch_size=10000,
        desc="Create association questions",
    )
    ds_ass_per = ds_ass_per.map(select_association_question)

    ds_ass_cap = ds_ass_cap.map(
        generate_association_question,
        fn_kwargs={"categories": categories, "use_persona": False},
        batched=True,
        batch_size=10000,
        desc="Create association questions",
    )
    ds_ass_cap = ds_ass_cap.map(select_association_question)

    print("Create consistency questions ...")
    ds_consist = ds_consist.map(
        generate_consistency_question,
        fn_kwargs={"categories": categories},
        batched=True,
        batch_size=10000,
        desc="Create consistency questions",
    )
    ds_consist = ds_consist.map(select_consistency_question)

    print("Assemble final dataset ...")
    ds_qa_prob = ds_qa_prob.select_columns(
        [
            "id",
            "type",
            "user",
            "assistant",
            "category_name",
            "category_key",
            "lang_key",
            "lang_name",
            "domain",
            "difficulty",
            "path_img",
            "code",
        ]
    )
    ds_ps_desc = ds_ps_desc.select_columns(
        [
            "id",
            "type",
            "user",
            "assistant",
            "category_name",
            "category_key",
            "lang_key",
            "lang_name",
            "domain",
            "difficulty",
            "path_img",
            "code",
        ]
    )
    ds_ps_capt = ds_ps_capt.select_columns(
        [
            "id",
            "type",
            "user",
            "assistant",
            "category_name",
            "category_key",
            "lang_key",
            "lang_name",
            "domain",
            "difficulty",
            "path_img",
            "code",
        ]
    )
    ds_ass_per = ds_ass_per.select_columns(
        [
            "id",
            "type",
            "user",
            "assistant",
            "category_name",
            "category_key",
            "lang_key",
            "lang_name",
            "domain",
            "difficulty",
            "path_img",
            "code",
        ]
    )
    ds_ass_cap = ds_ass_cap.select_columns(
        [
            "id",
            "type",
            "user",
            "assistant",
            "category_name",
            "category_key",
            "lang_key",
            "lang_name",
            "domain",
            "difficulty",
            "path_img",
            "code",
        ]
    )
    ds_consist = ds_consist.select_columns(
        [
            "id",
            "type",
            "user",
            "assistant",
            "category_name",
            "category_key",
            "lang_key",
            "lang_name",
            "domain",
            "difficulty",
            "path_img",
            "code",
        ]
    )
    ds_llm_gen = ds_llm_gen.select_columns(
        [
            "id",
            "type",
            "user",
            "assistant",
            "category_name",
            "category_key",
            "lang_key",
            "lang_name",
            "domain",
            "difficulty",
            "path_img",
            "code",
        ]
    )
    ds_structu = ds_structu.select_columns(
        [
            "id",
            "type",
            "user",
            "assistant",
            "category_name",
            "category_key",
            "lang_key",
            "lang_name",
            "domain",
            "difficulty",
            "path_img",
            "code",
        ]
    )
    ds_transla = ds_transla.select_columns(
        [
            "id",
            "type",
            "user",
            "assistant",
            "category_name",
            "category_key",
            "lang_key",
            "lang_name",
            "domain",
            "difficulty",
            "path_img",
            "code",
        ]
    )

    final_ds = concatenate_datasets(
        [ds_qa_prob, ds_ps_desc, ds_ps_capt, ds_ass_per, ds_ass_cap, ds_consist, ds_llm_gen, ds_structu, ds_transla]
    ).shuffle(seed=42)
    final_ds.to_json(f"{save_filepath}/dataset_final_assembled.jsonl")
    save_stats_train(final_ds, f"{save_filepath}/statistics_final.json")

    final_ds_half_1 = concatenate_datasets(
        [
            ds_qa_prob.shuffle(seed=42).select(range(len(ds_qa_prob) // 2)),
            ds_ps_desc.shuffle(seed=42).select(range(len(ds_ps_desc) // 2)),
            ds_ps_capt.shuffle(seed=42).select(range(len(ds_ps_capt) // 2)),
            ds_ass_per.shuffle(seed=42).select(range(len(ds_ass_per) // 2)),
            ds_ass_cap.shuffle(seed=42).select(range(len(ds_ass_cap) // 2)),
            ds_consist.shuffle(seed=42).select(range(len(ds_consist) // 2)),
            ds_llm_gen.shuffle(seed=42).select(range(len(ds_llm_gen) // 2)),
            ds_structu.shuffle(seed=42).select(range(len(ds_structu) // 2)),
            ds_transla.shuffle(seed=42).select(range(len(ds_transla) // 2)),
        ]
    ).shuffle(seed=42)
    final_ds_half_1.to_json(f"{save_filepath}/dataset_final_assembled_half_1.jsonl")
    save_stats_train(final_ds_half_1, f"{save_filepath}/statistics_final_half_1.json")

    final_ds_half_2 = concatenate_datasets(
        [
            ds_qa_prob.shuffle(seed=42).select(range(len(ds_qa_prob) // 2, len(ds_qa_prob))),
            ds_ps_desc.shuffle(seed=42).select(range(len(ds_ps_desc) // 2, len(ds_ps_desc))),
            ds_ps_capt.shuffle(seed=42).select(range(len(ds_ps_capt) // 2, len(ds_ps_capt))),
            ds_ass_per.shuffle(seed=42).select(range(len(ds_ass_per) // 2, len(ds_ass_per))),
            ds_ass_cap.shuffle(seed=42).select(range(len(ds_ass_cap) // 2, len(ds_ass_cap))),
            ds_consist.shuffle(seed=42).select(range(len(ds_consist) // 2, len(ds_consist))),
            ds_llm_gen.shuffle(seed=42).select(range(len(ds_llm_gen) // 2, len(ds_llm_gen))),
            ds_structu.shuffle(seed=42).select(range(len(ds_structu) // 2, len(ds_structu))),
            ds_transla.shuffle(seed=42).select(range(len(ds_transla) // 2, len(ds_transla))),
        ]
    ).shuffle(seed=42)
    final_ds_half_2.to_json(f"{save_filepath}/dataset_final_assembled_half_2.jsonl")
    save_stats_train(final_ds_half_2, f"{save_filepath}/statistics_final_half_2.json")


if __name__ == "__main__":
    main()
