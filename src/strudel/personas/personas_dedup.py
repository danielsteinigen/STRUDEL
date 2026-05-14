from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromFileSystem, MinHashDedup

with Pipeline(
    name="personas-dedup",
) as pipeline:
    input_batch_size = 10000

    data_loader = LoadDataFromFileSystem(
        data_files="/data/personas/personas_9M.jsonl",
        split="train",
        streaming=True,
        batch_size=input_batch_size,
    )

    minhash_dedup = MinHashDedup(
        tokenizer="words",
        storage="dict",
        seed=1,
        threshold=0.9,
        input_batch_size=input_batch_size,
        input_mappings={"text": "persona"},
    )

    data_loader >> minhash_dedup


if __name__ == "__main__":

    distiset = pipeline.run(use_cache=False)
    if distiset:
        ds = distiset["default"]["train"]
        ds_dedup = ds.filter(lambda x: x["keep_row_after_minhash_filtering"], num_proc=8).select_columns(["id", "persona"])
        ds_dedup.to_json(f"/data/personas/personas_dedup.jsonl")
        ds_dup = ds.filter(lambda x: not x["keep_row_after_minhash_filtering"], num_proc=8).select_columns(["id", "persona"])
        ds_dup.to_json(f"/data/personas/personas_dup.jsonl")
