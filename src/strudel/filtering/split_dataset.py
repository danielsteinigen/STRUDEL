import argparse
import json
import os
import random
from collections import Counter

from datasets import Dataset, concatenate_datasets, load_dataset
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from strudel.util import load_json, save_json

random.seed(42)
save_filepath = f"/data/dataset_v10/dataset_assembled"


def save_stats(ds, path):
    print("Save statistics ...")
    cat_counts = Counter(ds["category_key"])
    lang_counts = Counter(ds["cat_lang"])
    compl_counts = Counter(ds["cat_lang_compl"])
    stats = {
        "total_count": len(ds),
        "category_counts": dict(cat_counts),
        "language_counts": dict(lang_counts),
        "complexity_counts": dict(compl_counts),
    }
    save_json(path, stats)


def get_subset_qa(ds: Dataset):
    category_col = "cat_lang"
    categories = ds.unique(category_col)

    shards_1 = []
    shards_2 = []
    shards_3 = []
    shards_4 = []
    for cat in categories:
        bucket = ds.filter(lambda ex: ex[category_col] == cat).shuffle(seed=42)
        buck_len = len(bucket)
        if cat in ["quantum_qasm", "mol_smiles"]:
            bucket_1 = bucket.select(range(4000))
            bucket_2 = bucket.select(range(buck_len - 1000, buck_len))
            bucket = bucket.shuffle(seed=42)
            bucket_3 = bucket.select(range(1000))
            bucket_4 = bucket.select(range(0))
        elif cat in ["dna_vienna", "chess_fen"]:
            bucket_1 = bucket.select(range(buck_len))
            bucket_2 = bucket.select(range(buck_len))
            bucket_3 = bucket.select(range(buck_len))
            bucket_4 = bucket.select(range(0))
        elif cat in ["chart_plotly"]:
            bucket_stat = bucket.filter(lambda sample: json.loads(sample["statistics"])["node_types"])
            bucket_not_stat = bucket.filter(lambda sample: not json.loads(sample["statistics"])["node_types"])
            bucket = concatenate_datasets([bucket_stat.select(range(2000, len(bucket_stat))), bucket_not_stat]).shuffle(seed=42)
            bucket_1 = bucket.select(range(4000))
            bucket_2 = bucket.select(range(4000, 5000))
            bucket_3 = bucket.select(range(5000, 6000))
            bucket_4 = concatenate_datasets([bucket_stat.select(range(2000)), bucket.select(range(6000, len(bucket)))]).shuffle(seed=42)
        else:
            bucket_1 = bucket.select(range(4000))
            bucket_2 = bucket.select(range(4000, 5000))
            bucket_3 = bucket.select(range(5000, 6000))
            bucket_4 = bucket.select(range(6000, len(bucket)))
        shards_1.append(bucket_1)
        shards_2.append(bucket_2)
        shards_3.append(bucket_3)
        shards_4.append(bucket_4)

    out_1 = concatenate_datasets(shards_1).shuffle(seed=42)
    out_2 = concatenate_datasets(shards_2).shuffle(seed=42)
    out_3 = concatenate_datasets(shards_3).shuffle(seed=42)
    out_4 = concatenate_datasets(shards_4).shuffle(seed=42)
    return out_1, out_2, out_3, out_4


def get_subset_ps(ds: Dataset):
    category_col = "cat_lang"
    categories = ds.unique(category_col)

    shards_1 = []
    shards_2 = []
    shards_3 = []
    shards_4 = []
    for cat in categories:
        bucket = ds.filter(lambda ex: ex[category_col] == cat).shuffle(seed=42)
        buck_len = len(bucket)
        if cat in ["quantum_qasm", "dna_fasta", "dna_vienna", "mol_smiles", "mol_smarts_react", "neural_nn_onnx_graph"]:
            bucket_1 = bucket.select(range(2000))
            bucket_2 = bucket.select(range(buck_len - 2000, buck_len))
            bucket = bucket.shuffle(seed=42)
            bucket_3 = bucket.select(range(2000))
            bucket_4 = bucket.select(range(0))
        elif cat in ["chess_fen"]:
            bucket_1 = bucket.select(range(buck_len))
            bucket_2 = bucket.select(range(buck_len))
            bucket_3 = bucket.select(range(buck_len))
            bucket_4 = bucket.select(range(0))
        elif cat in ["chart_plotly"]:
            bucket_stat = bucket.filter(lambda sample: json.loads(sample["statistics"])["node_types"])
            bucket_not_stat = bucket.filter(lambda sample: not json.loads(sample["statistics"])["node_types"])
            bucket = concatenate_datasets([bucket_stat.select(range(2000, len(bucket_stat))), bucket_not_stat]).shuffle(seed=42)
            bucket_1 = bucket.select(range(2000))
            bucket_2 = bucket.select(range(2000, 4000))
            bucket_3 = bucket.select(range(4000, 6000))
            bucket_4 = concatenate_datasets([bucket_stat.select(range(2000)), bucket.select(range(6000, len(bucket)))]).shuffle(seed=42)
        else:
            bucket_1 = bucket.select(range(2000))
            bucket_2 = bucket.select(range(2000, 4000))
            bucket_3 = bucket.select(range(4000, 6000))
            bucket_4 = bucket.select(range(6000, len(bucket)))
        shards_1.append(bucket_1)
        shards_2.append(bucket_2)
        shards_3.append(bucket_3)
        shards_4.append(bucket_4)

    out_1 = concatenate_datasets(shards_1).shuffle(seed=42)
    out_2 = concatenate_datasets(shards_2).shuffle(seed=42)
    out_3 = concatenate_datasets(shards_3).shuffle(seed=42)
    out_4 = concatenate_datasets(shards_4).shuffle(seed=42)
    return out_1, out_2, out_3, out_4


def get_subset_struct(ds_false: Dataset, ds_rest: Dataset):
    category_col = "cat_lang"
    categories_false = ds_false.unique(category_col)
    categories_rest = ds_rest.unique(category_col)
    categories = list(set(categories_false) | set(categories_rest))
    n_target = 2000

    shards_1 = []
    shards_2 = []
    shards_3 = []
    for cat in categories:
        bucket_false = ds_false.filter(lambda ex: ex[category_col] == cat).shuffle(seed=42)
        bucket_rest = ds_rest.filter(lambda ex: ex[category_col] == cat).shuffle(seed=42)

        if len(bucket_rest) > (n_target * 3):
            if cat in ["chart_plotly"]:
                bucket_rest_stat = bucket_rest.filter(lambda sample: json.loads(sample["statistics"])["node_types"])
                bucket_rest_not_stat = bucket_rest.filter(lambda sample: not json.loads(sample["statistics"])["node_types"])
                bucket_rest = concatenate_datasets(
                    [bucket_rest_stat.select(range(n_target, len(bucket_rest_stat))), bucket_rest_not_stat]
                ).shuffle(seed=42)
                bucket_1 = bucket_rest_stat.select(range(n_target))
            else:
                bucket_1 = bucket_rest.select(range(n_target))
            bucket_2 = bucket_rest.select(range(n_target, (2 * n_target)))
            bucket_3 = bucket_rest.select(range((2 * n_target), (3 * n_target)))
        else:
            buck_len = len(bucket_rest) // 3
            bucket_1 = concatenate_datasets([bucket_rest.select(range(buck_len)), bucket_false.select(range(n_target - buck_len))]).shuffle(
                seed=42
            )
            bucket_2 = concatenate_datasets(
                [bucket_rest.select(range(buck_len, (2 * buck_len))), bucket_false.select(range(n_target - buck_len))]
            ).shuffle(seed=42)
            bucket_3 = concatenate_datasets(
                [bucket_rest.select(range((2 * buck_len), (3 * buck_len))), bucket_false.select(range(n_target - buck_len))]
            ).shuffle(seed=42)

        shards_1.append(bucket_1)
        shards_2.append(bucket_2)
        shards_3.append(bucket_3)

    out_1 = concatenate_datasets(shards_1).shuffle(seed=42)
    out_2 = concatenate_datasets(shards_2).shuffle(seed=42)
    out_3 = concatenate_datasets(shards_3).shuffle(seed=42)
    return out_1, out_2, out_3


def create_statistics_rdkit(sample):
    if sample["lang_key"] != "smiles":
        return {"statistics": sample["statistics"]}
    else:
        mol = Chem.MolFromSmiles(sample["code"])
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_counts = Counter(atom_symbols)
        renamed_counts = {f"{el} atoms": count for el, count in atom_counts.items()}
        renamed_counts["rings"] = rdMolDescriptors.CalcNumRings(mol)
        return {"statistics": json.dumps({"node_types": renamed_counts}, ensure_ascii=False)}


def main():
    global save_filepath

    parser = argparse.ArgumentParser(description="Split filtered datasets into refinement subsets.")
    parser.add_argument("--qa-input-dirs", nargs="+", required=True, help="Directories containing QA-scored datasets")
    parser.add_argument("--ps-input-dirs", nargs="+", required=True, help="Directories containing PS-scored datasets")
    parser.add_argument("--output-dir", required=True, help="Directory where split datasets are written")
    parser.add_argument("--categories", default="diagram_categories.json", help="Path to category definitions JSON")
    args = parser.parse_args()

    save_filepath = args.output_dir
    os.makedirs(save_filepath, exist_ok=True)

    categories = load_json(filename=args.categories)
    save_dirs_qa = args.qa_input_dirs
    save_dirs_ps = args.ps_input_dirs
    ds_qa = load_dataset("json", data_files=[f"{save_dir}/dataset_score_final.jsonl" for save_dir in save_dirs_qa])["train"].shuffle(
        seed=42
    )
    ds_ps = load_dataset("json", data_files=[f"{save_dir}/dataset_score_final.jsonl" for save_dir in save_dirs_ps])["train"].shuffle(
        seed=42
    )
    ds_qa_false = load_dataset("json", data_files=[f"{save_dir}/dataset_score_final_false.jsonl" for save_dir in save_dirs_qa])[
        "train"
    ].shuffle(seed=42)
    ds_ps_false = load_dataset("json", data_files=[f"{save_dir}/dataset_score_final_false.jsonl" for save_dir in save_dirs_ps])[
        "train"
    ].shuffle(seed=42)

    print("Renaming and statistics")
    ds_qa = ds_qa.rename_column("path_img_1", "path_img")
    ds_ps = ds_ps.rename_column("path_img_1", "path_img")
    ds_qa_false = ds_qa_false.rename_column("path_img_1", "path_img")
    ds_ps_false = ds_ps_false.rename_column("path_img_1", "path_img")
    ds_qa = ds_qa.map(create_statistics_rdkit)
    ds_ps = ds_ps.map(create_statistics_rdkit)
    ds_qa_false = ds_qa_false.map(create_statistics_rdkit)
    ds_ps_false = ds_ps_false.map(create_statistics_rdkit)

    print("Generate combined set")
    ds_comb = concatenate_datasets([ds_qa, ds_ps]).shuffle(seed=42)
    ds_comb.to_json(f"{save_filepath}/dataset_combined.jsonl")
    save_stats(ds_comb, f"{save_filepath}/statistics_combined.json")
    ds_comb_false = concatenate_datasets([ds_qa_false, ds_ps_false]).shuffle(seed=42)
    ds_comb_false.to_json(f"{save_filepath}/dataset_combined_false.jsonl")
    save_stats(ds_comb_false, f"{save_filepath}/statistics_combined_false.json")

    print("Generate QA subset")
    ds_qa_prob, ds_ass_per, ds_ass_cap, ds_qa_rest = get_subset_qa(ds_qa)
    print("Generate PS subset")
    ds_ps_desc, ds_ps_capt, ds_consist, ds_ps_rest = get_subset_ps(ds_ps)

    print("Generate QA struct")
    ds_qa_structu, ds_qa_llm_gen, ds_qa_transla = get_subset_struct(ds_qa_false, ds_qa_rest)
    print("Generate PS struct")
    ds_ps_structu, ds_ps_llm_gen, ds_ps_transla = get_subset_struct(ds_ps_false, ds_ps_rest)

    print("Concat sets and save")
    ds_structu = concatenate_datasets([ds_qa_structu, ds_ps_structu])
    ds_llm_gen = concatenate_datasets([ds_qa_llm_gen, ds_ps_llm_gen])
    ds_transla = concatenate_datasets([ds_qa_transla, ds_ps_transla])

    ds_qa_prob.shuffle(seed=42).to_json(f"{save_filepath}/dataset_qa_problem.jsonl")
    ds_ps_desc.shuffle(seed=42).to_json(f"{save_filepath}/dataset_ps_description.jsonl")
    ds_ps_capt.shuffle(seed=42).to_json(f"{save_filepath}/dataset_ps_caption.jsonl")
    ds_ass_per.shuffle(seed=42).to_json(f"{save_filepath}/dataset_association_persona.jsonl")
    ds_ass_cap.shuffle(seed=42).to_json(f"{save_filepath}/dataset_association_caption.jsonl")
    ds_consist.shuffle(seed=42).to_json(f"{save_filepath}/dataset_consistency.jsonl")
    ds_llm_gen.shuffle(seed=42).to_json(f"{save_filepath}/dataset_llm_qa_gen.jsonl")
    ds_structu.shuffle(seed=42).to_json(f"{save_filepath}/dataset_structural.jsonl")
    ds_transla.shuffle(seed=42).to_json(f"{save_filepath}/dataset_code_translate.jsonl")
    save_stats(ds_qa_prob, f"{save_filepath}/statistics_qa_problem.json")
    save_stats(ds_ps_desc, f"{save_filepath}/statistics_ps_description.json")
    save_stats(ds_ps_capt, f"{save_filepath}/statistics_ps_caption.json")
    save_stats(ds_ass_per, f"{save_filepath}/statistics_association_persona.json")
    save_stats(ds_ass_cap, f"{save_filepath}/statistics_association_caption.json")
    save_stats(ds_consist, f"{save_filepath}/statistics_consistency.json")
    save_stats(ds_llm_gen, f"{save_filepath}/statistics_llm_qa_gen.json")
    save_stats(ds_structu, f"{save_filepath}/statistics_structural.json")
    save_stats(ds_transla, f"{save_filepath}/statistics_code_translate.json")


if __name__ == "__main__":
    main()
