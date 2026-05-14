import argparse
import json
import os
import shutil
from collections import Counter, defaultdict

import imagehash
from datasets import Image, load_dataset
from PIL import Image as PIL_Image

from strudel.filtering.mappings import get_map_key, map_domain, selected_categories_less
from strudel.util import classify_image_black_or_white, load_json, save_json

# T10: Question-Answer(QA), T5: Problem-Solution(PS)
save_filepath = "/data/data/strudel/datasets_test/test/dataset_ps_filtered_additional"
copy_images = False
max_dups_code = 3
max_dups_img = 3


def ensure_output_dirs():
    os.makedirs(save_filepath, exist_ok=True)
    if copy_images:
        os.makedirs(f"{save_filepath}/img_additional/", exist_ok=True)
        os.makedirs(f"{save_filepath}/img_ratio/", exist_ok=True)
        os.makedirs(f"{save_filepath}/img_stats/", exist_ok=True)
        os.makedirs(f"{save_filepath}/img_duplicates/", exist_ok=True)


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


def additional_filter(sample):
    keep = True
    if classify_image_black_or_white(sample["image"], threshold=0.992):
        keep = False
    if sample["cat_lang"] == "dna_vienna" and not str(sample["path_img_2"]).strip():
        keep = False
    if not keep and copy_images:
        shutil.copy(sample["path_img_1"], f"{save_filepath}/img_additional/{os.path.basename(sample['path_img_1'])}")
    return keep


def aspect_ratio_filter(sample, max_ratio=3):
    if sample["category_key"] in ["music", "dna"]:
        max_ratio = 7
    elif sample["category_key"] in ["neural", "gantt"]:
        max_ratio = 4
    elif sample["category_key"] in ["quantum", "logic"]:
        max_ratio = 3.5
    w, h = sample["size"]["width"], sample["size"]["height"]
    if h == 0:
        return False
    ratio = w / h
    keep = (1 / max_ratio) <= ratio <= max_ratio
    if not keep and copy_images:
        shutil.copy(sample["path_img_1"], f"{save_filepath}/img_ratio/{os.path.basename(sample['path_img_1'])}")
    return keep


def check_limit(node_types, code, node, thresh, mode):
    if not node:
        return True
    if node == "all":
        return all(x <= thresh if mode == "max" else x >= thresh for x in node_types.values())
    if node == "sum":
        total = sum(node_types.values())
        return total <= thresh if mode == "max" else total >= thresh
    if node == "len":
        length = len(code.strip())
        return length <= thresh if mode == "max" else length >= thresh
    return node in node_types and (node_types[node] <= thresh if mode == "max" else node_types[node] >= thresh)


def stats_filter(sample, categories):
    category = sample["category_key"]
    lang_key = sample["lang_key"]
    node_types = json.loads(sample["statistics"])["node_types"]
    if not node_types and category == "chart":
        return True
    if not node_types and category != "mol":
        return False
    if lang_key == "nn_onnx_graph" and "op_types" in node_types:
        node_types = node_types["op_types"]

    stats_limits = categories[category]["language"][lang_key].get("stats_limits", None)
    if not stats_limits:
        return True
    if lang_key == "fasta":
        keep_max = ((sum(node_types.values()) - node_types["sequences"]) / node_types["sequences"]) <= stats_limits.get("max_thresh")
    else:
        keep_max = check_limit(node_types, sample["code"], stats_limits.get("max_node"), stats_limits.get("max_thresh"), "max")
    keep_min = check_limit(node_types, sample["code"], stats_limits.get("min_node"), stats_limits.get("min_thresh"), "min")
    if not (keep_max and keep_min) and copy_images:
        shutil.copy(sample["path_img_1"], f"{save_filepath}/img_stats/{os.path.basename(sample['path_img_1'])}")
    return keep_max and keep_min


def duplicate_filter(ds, log_path, img=False):
    print("Start duplicate filtering ...")
    key_map = defaultdict(list)  # (category, content) -> [(idx, id), ...]
    for i, row in enumerate(ds):
        cid = row.get("path_img_1", "")  # if img else row.get("id", str(i))
        content = row["image_hash"] if img else "".join(row["code"].split())
        key_map[(row["cat_lang"], content)].append((i, cid))

    if copy_images:
        per_cat = defaultdict(list)  # category -> [(content, count, [ids])]
        for (cat, content), pairs in key_map.items():
            if len(pairs) > 1:
                per_cat[cat].append((content, len(pairs), [id_ for _, id_ in pairs]))

        with open(f"{log_path}/{'img' if img else 'code'}_duplicates.txt", "w") as text_file:
            for cat, items in per_cat.items():
                items.sort(key=lambda x: x[1], reverse=True)
                text_file.write(f"\n\nCategory: {cat}\n")
                for rank, (code, cnt, ids) in enumerate(items, 1):
                    preview = (code[:120] + "…") if len(code) > 120 else code
                    text_file.write(f"  #{rank} dup_count={cnt}\n")
                    if not img:
                        text_file.write(f"      code_preview: {preview!r}\n")
                    text_file.write(f"      ids: {ids}\n")

                    check_path = f"{log_path}/img_duplicates/{'img' if img else 'code'}_{cat}_{rank}/"
                    os.makedirs(check_path, exist_ok=True)
                    for pth in ids:
                        shutil.copy(pth, f"{check_path}/{os.path.basename(pth)}")
        del per_cat

    max_dups = max_dups_code if not img else max_dups_img
    keep_idx = sorted(i for pairs in key_map.values() for i, _ in pairs[:max_dups])
    del key_map
    keep_idx_set = set(keep_idx)
    remove_idx = [i for i in range(len(ds)) if i not in keep_idx_set]

    print("Done duplicate filtering.")
    return keep_idx, remove_idx


def main():
    global save_filepath, copy_images, max_dups_code, max_dups_img

    parser = argparse.ArgumentParser(description="Filter generated datasets.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Directories containing dataset.jsonl files")
    parser.add_argument("--output-dir", required=True, help="Directory where filtered output is written")
    parser.add_argument("--categories", default="diagram_categories.json", help="Path to category definitions JSON")
    parser.add_argument("--copy-images", action="store_true", help="Copy rejected samples into inspection folders")
    parser.add_argument("--max-dups-code", type=int, default=3, help="Maximum duplicate code samples to keep")
    parser.add_argument("--max-dups-img", type=int, default=3, help="Maximum duplicate image samples to keep")
    args = parser.parse_args()

    save_filepath = args.output_dir
    copy_images = args.copy_images
    max_dups_code = args.max_dups_code
    max_dups_img = args.max_dups_img
    ensure_output_dirs()

    categories = load_json(filename=args.categories)

    save_dirs = args.input_dirs

    print("Convert statistics to string")
    for save_dir in save_dirs:
        inp = f"{save_dir}/dataset.jsonl"
        out = f"{save_dir}/dataset_str.jsonl"
        with open(inp, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
            for line in fin:
                ex = json.loads(line)
                ex["statistics"] = json.dumps(ex.get("statistics", {}), ensure_ascii=False)
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    ds = load_dataset(
        "json",
        data_files=[f"{save_dir}/dataset_str.jsonl" for save_dir in save_dirs],
    )["train"]

    print("Filter out categories that are not selected ...")
    ds = ds.map(lambda sample: {"category_key": f"{sample['input']['category_key']}"})
    ds = ds.map(lambda sample: {"category_name": f"{sample['input']['category_name']}"})
    ds = ds.map(lambda sample: {"lang_key": f"{sample['input']['lang_key']}"})
    ds = ds.map(lambda sample: {"lang_name": f"{sample['input']['lang_name']}"})
    ds = ds.filter(
        lambda sample: sample["category_key"] in selected_categories_less
        and sample["lang_key"] in selected_categories_less[sample["category_key"]]
    )
    ds = ds.map(lambda sample: {"cat_lang": f"{sample['category_key']}_{sample['lang_key']}"})
    ds = ds.map(lambda sample: {"cat_lang_compl": f"{sample['cat_lang']}_{sample['input']['complexity']}"})
    ds = ds.map(lambda sample: {"domain": map_domain.get(get_map_key(sample["category_key"], sample["lang_key"]), "")})
    save_stats(ds, f"{save_filepath}/stats_filter_0_stop.json")

    print("Filter out invalid images ...")
    ds = ds.filter(lambda sample: str(sample["path_img_1"]).strip())
    save_stats(ds, f"{save_filepath}/stats_filter_1_invalid.json")

    print("Filter out to many nodes or to less/no nodes based on statistics ...")
    ds = ds.filter(stats_filter, fn_kwargs={"categories": categories})
    save_stats(ds, f"{save_filepath}/stats_filter_2_nodecnt.json")

    print("Filter aspect ratio ...")
    ds = ds.filter(aspect_ratio_filter)
    save_stats(ds, f"{save_filepath}/stats_filter_3_ratio.json")

    print("Filter duplicates code ...")
    keep_idx, remove_idx = duplicate_filter(ds, save_filepath)
    ds_dup = ds.select(remove_idx)
    ds = ds.select(keep_idx)
    save_stats(ds, f"{save_filepath}/stats_filter_4_dupcode.json")
    del keep_idx
    del remove_idx

    print("Filter mainly white images + vienna image 2 + keras image 2 ...")
    ds = ds.map(lambda sample: {"image": sample["path_img_1"]})
    ds = ds.cast_column("image", Image(mode="RGB"))
    ds = ds.filter(additional_filter)
    save_stats(ds, f"{save_filepath}/stats_filter_5_additional.json")

    print("Filter duplicates images ...")
    ds = ds.map(lambda sample: {"image_hash": str(imagehash.phash(sample["image"]))})
    img_keep_idx, img_remove_idx = duplicate_filter(ds, save_filepath, img=True)
    ds_dup_img = ds.select(img_remove_idx)
    ds = ds.select(img_keep_idx)
    save_stats(ds, f"{save_filepath}/stats_filter_6_dupimg.json")
    del img_keep_idx
    del img_remove_idx

    print("Save dataset ...")
    ds = ds.remove_columns(["image"])
    ds.to_json(f"{save_filepath}/dataset.jsonl")
    ds_dup.to_json(f"{save_filepath}/dataset_dub.jsonl")
    ds_dup_img.to_json(f"{save_filepath}/dataset_dub_img.jsonl")


if __name__ == "__main__":
    main()
