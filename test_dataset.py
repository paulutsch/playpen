from datasets import ClassLabel, concatenate_datasets, load_dataset

playpen_dataset = load_dataset(
    "clembench-playpen/DPO_turn-level_10Klimit", split="train"
)
playpen_dataset_train = playpen_dataset.train_test_split(0.2, shuffle=True, seed=42)

tulu_dataset = load_dataset(
    "allenai/llama-3.1-tulu-3-8b-preference-mixture", split="train"
)
tulu_sub_ratio = len(playpen_dataset_train["train"]) / len(tulu_dataset)
unique_sources = tulu_dataset.unique("source")
source_classlabel = ClassLabel(names=unique_sources)  # needed for stratified split
tulu_dataset = tulu_dataset.cast_column("source", source_classlabel)

tulu_split = tulu_dataset.train_test_split(
    train_size=tulu_sub_ratio,
    stratify_by_column="source",
    seed=8,  # tulu3 uses seed 8 for SFT too
)

tulu_sub_dataset = tulu_split["train"]

assert len(tulu_sub_dataset) == len(
    playpen_dataset["train"]
), f"Length mismatch: tulu_sub_dataset={len(tulu_sub_dataset)}, clembench dataset={len(playpen_dataset['train'])}"

combined_dataset = concatenate_datasets([playpen_dataset["train"], tulu_sub_dataset])
print(f"Size of the train set: {len(combined_dataset)}")
