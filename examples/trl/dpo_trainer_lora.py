import json
import os
import random
from collections import Counter, defaultdict

import trl
from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry
from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig

from playpen import BasePlayPen

# For wandb api-key
with open("key.json", "r") as f:
    keys = json.load(f)

os.environ["WANDB_API_KEY"] = keys["wandb"]["api_key"]
os.environ["WANDB_PROJECT"] = "llama3-dpo"


def as_key(prompt):
    # Canonical JSON → stable, hashable string
    return json.dumps(prompt, sort_keys=True, ensure_ascii=False)


class PeftDpoTrainer(BasePlayPen):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        # Note: We configure the proper chat template for the tokenizer already during model loading in the backend

    def learn(self, game_registry: GameRegistry):
        # Load a preference dataset for DPO, that is, a list of preference pairs.
        # Each example should have "prompt", "chosen", and "rejected" fields.
        # During training, the model learns to assign higher rewards to chosen responses and lower rewards to rejected responses.

        # --- Dataset loading ---

        playpen_dataset = load_dataset(
            "clembench-playpen/DPO_turn-level_10Klimit", split="train"
        )
        playpen_dataset = playpen_dataset.train_test_split(0.2, shuffle=True, seed=42)

        def balanced_subsample(ds, max_per_prompt=100):
            keys = defaultdict(list)
            for i, ex in enumerate(ds):
                keys[as_key(ex["prompt"])].append(i)
            keep = [
                idx
                for indices in keys.values()
                for idx in random.sample(indices, min(len(indices), max_per_prompt))
            ]
            return ds.select(keep)

        playpen_dataset["train"] = balanced_subsample(
            playpen_dataset["train"], max_per_prompt=20
        )

        tulu_dataset = load_dataset(
            "allenai/llama-3.1-tulu-3-8b-preference-mixture", split="train"
        )
        tulu_sub_ratio = len(playpen_dataset["train"]) / len(tulu_dataset)
        unique_sources = tulu_dataset.unique("source")
        source_classlabel = ClassLabel(
            names=unique_sources
        )  # needed for stratified split
        tulu_dataset = tulu_dataset.cast_column("source", source_classlabel)

        tulu_split = tulu_dataset.train_test_split(
            train_size=tulu_sub_ratio,
            stratify_by_column="source",
            seed=8,  # tulu3 uses seed 8 for SFT too
        )

        tulu_sub_dataset = tulu_split["train"]

        def report_unique_prompts(playpen_ds, label="playpen"):
            """
            Count distinct `prompt` fields *before* they get merged into chosen/rejected.
            Also report the total number of samples in each split.
            """

            # Unique-prompt sets
            train_keys = {as_key(ex["prompt"]) for ex in playpen_ds["train"]}
            test_keys = {as_key(ex["prompt"]) for ex in playpen_ds["test"]}
            all_keys = train_keys | test_keys

            # Raw sample totals
            n_train = len(playpen_ds["train"])
            n_test = len(playpen_ds["test"])
            n_total = n_train + n_test

            print(f"\n🔍  Prompt census for '{label}':")
            print("──────────────────────────────────────────")
            print(
                f"  • train split : {len(train_keys):>6} unique  | {n_train:>6} total"
            )
            print(f"  • test  split : {len(test_keys):>6} unique  | {n_test:>6} total")
            print("  ────────────────────────────────────────")
            print(
                f"  • overall     : {len(all_keys):>6} unique  | {n_total:>6} total\n"
            )

            # Return everything in case downstream code wants it
            return {
                "unique": {"train": train_keys, "test": test_keys, "total": all_keys},
                "counts": {"train": n_train, "test": n_test, "total": n_total},
            }

        report_unique_prompts(playpen_dataset, label="playpen (pre-conversion)")

        # convert datasets to DPO format (no prompt field needed)
        def convert_playpen_to_dpo_format(example):
            # playpen: combine prompt + chosen/rejected, then remove prompt field
            if "prompt" in example:
                example["chosen"] = [*example["prompt"], *example["chosen"]]
                example["rejected"] = [*example["prompt"], *example["rejected"]]
                del example["prompt"]

            return example

        def convert_tulu_to_dpo_format(example):
            # tulu: remove prompt field
            del example["prompt"]
            return example

        print("=== PLAYPEN DATASET EXAMPLES (before conversion) ===")
        for i in range(2):
            print(f"\nExample {i+1} (chosen):")
            print(json.dumps(playpen_dataset["train"][i]["prompt"], indent=2))
            print(json.dumps(playpen_dataset["train"][i]["chosen"], indent=2))
            print(f"\nExample {i+1} (rejected):")
            print(json.dumps(playpen_dataset["train"][i]["prompt"], indent=2))
            print(json.dumps(playpen_dataset["train"][i]["rejected"], indent=2))

        # Apply conversion to both datasets
        tulu_sub_dataset = tulu_sub_dataset.map(convert_tulu_to_dpo_format)
        playpen_dataset["train"] = playpen_dataset["train"].map(
            convert_playpen_to_dpo_format
        )
        playpen_dataset["test"] = playpen_dataset["test"].map(
            convert_playpen_to_dpo_format
        )

        def show_multi_turn_example(
            dataset, label="playpen-train", min_len=3, scan_limit=2_000
        ):
            """
            Print the first row whose `chosen` list length >= `min_len`.
            Stops after `scan_limit` rows to avoid a pilgrimage through the entire set.
            """
            for idx, row in enumerate(dataset):
                if idx >= scan_limit:
                    break
                if len(row["chosen"]) >= min_len:
                    print(
                        f"\n=== {label.upper()} MULTI-TURN EXAMPLE #{idx} "
                        f"(messages per side: {len(row['chosen'])}) ==="
                    )

                    print("\n👁️  CHOSEN:")
                    print(json.dumps(row["chosen"], indent=2, ensure_ascii=False))

                    print("\n💔  REJECTED:")
                    print(json.dumps(row["rejected"], indent=2, ensure_ascii=False))
                    return

            print(
                f"\n🤷  Couldn’t find an example with ≥{min_len} messages "
                f"in the first {scan_limit} rows of '{label}'."
            )

        show_multi_turn_example(playpen_dataset["train"], label="playpen-train")

        chosen_lengths = [
            len(example["chosen"]) for example in playpen_dataset["train"]
        ]
        rejected_lengths = [
            len(example["rejected"]) for example in playpen_dataset["train"]
        ]

        # Create 10 bins for distribution analysis
        min_len = min(min(chosen_lengths), min(rejected_lengths))
        max_len = max(max(chosen_lengths), max(rejected_lengths))
        bin_size = (max_len - min_len) / 10

        print(f"\n=== MESSAGE COUNT DISTRIBUTION (10 bins) ===")
        print(f"Range: {min_len} to {max_len} messages")
        print(f"Bin size: {bin_size:.1f}")

        for i in range(1):
            bin_start = min_len + i * bin_size
            bin_end = min_len + (i + 1) * bin_size

            chosen_in_bin = sum(1 for l in chosen_lengths if bin_start <= l < bin_end)
            rejected_in_bin = sum(
                1 for l in rejected_lengths if bin_start <= l < bin_end
            )

            print(
                f"Bin {i+1:2d}: [{bin_start:3.0f}, {bin_end:3.0f}) - Chosen: {chosen_in_bin:3d}, Rejected: {rejected_in_bin:3d}"
            )

        print(
            f"\nTotal samples - Chosen: {len(chosen_lengths)}, Rejected: {len(rejected_lengths)}"
        )

        print("=== PLAYPEN DATASET EXAMPLES ===")
        for i in range(2):
            print(f"\nExample {i+1} (chosen):")
            print(json.dumps(playpen_dataset["train"][i]["chosen"], indent=2))
            print(f"\nExample {i+1} (rejected):")
            print(json.dumps(playpen_dataset["train"][i]["rejected"], indent=2))

        print("\n=== TULU DATASET EXAMPLES ===")
        for i in range(2):
            print(f"\nExample {i+1} (chosen):")
            print(json.dumps(tulu_sub_dataset[i]["chosen"], indent=2))
            print(f"\nExample {i+1} (rejected):")
            print(json.dumps(tulu_sub_dataset[i]["rejected"], indent=2))

        assert len(tulu_sub_dataset) == len(
            playpen_dataset["train"]
        ), f"Length mismatch: tulu_sub_dataset={len(tulu_sub_dataset)}, clembench dataset={len(playpen_dataset['train'])}"

        def audit_prefix_integrity(dataset, name="dataset", max_print=5):
            """
            Scan every preference pair and bucket failures into three categories:
              A. length mismatch
              B. divergent prefix (i.e., chosen[:-1] != rejected[:-1])
              C. identical final assistant message (chosen[-1] == rejected[-1])
            Prints a short crime report and returns a Counter with the tallies.
            """

            crimes = Counter()  # keeps score so we can surface totals
            first_offenders = {  # store the first few indices for each crime
                "length": [],
                "prefix": [],
                "final_id": [],
            }

            for idx, row in enumerate(dataset):
                chosen, rejected = row["chosen"], row["rejected"]

                # A · Same length?
                if len(chosen) != len(rejected):
                    crimes["length"] += 1
                    if len(first_offenders["length"]) < max_print:
                        first_offenders["length"].append(idx)
                    continue  # no point in deeper checks—structure already broken

                # B · Identical prefix?
                if chosen[:-1] != rejected[:-1]:
                    crimes["prefix"] += 1
                    if len(first_offenders["prefix"]) < max_print:
                        first_offenders["prefix"].append(idx)
                    continue

                # C · Final answers must differ
                if chosen[-1] == rejected[-1]:
                    crimes["final_id"] += 1
                    if len(first_offenders["final_id"]) < max_print:
                        first_offenders["final_id"].append(idx)

            clean = len(dataset) - sum(crimes.values())

            print(f"\n🚨 Integrity audit for '{name}'")
            print("──────────────────────────────────────────")
            print(f"Total rows examined : {len(dataset):>6}")
            print(f"Perfectly clean     : {clean:>6}")
            print(f"Length mismatch     : {crimes['length']:>6}")
            print(f"Divergent prefix    : {crimes['prefix']:>6}")
            print(f"Identical finals    : {crimes['final_id']:>6}")
            print("──────────────────────────────────────────")

            return crimes

        audit_prefix_integrity(tulu_sub_dataset, name="tulu-sub")
        audit_prefix_integrity(playpen_dataset["train"], name="playpen-train")
        audit_prefix_integrity(playpen_dataset["test"], name="playpen-test")

        combined_dataset = concatenate_datasets(
            [playpen_dataset["train"], tulu_sub_dataset]
        )
        print(f"Size of the train set: {len(combined_dataset)}")

        # Initialize training configuration for dpo
        config = trl.DPOConfig(
            max_length=4096,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            num_train_epochs=2,
            lr_scheduler_type="linear",
            warmup_ratio=0.03,
            bf16=True,
            gradient_checkpointing=True,
            seed=7331,
            output_dir=f"models/dpo+lora/llama3-8b",
            logging_steps=10,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            # dpo-specific params
            beta=0.1,  # temperature for dpo loss — Tülu uses beta=5, but they use length-normalized DPO instead of a sum-level loss
            # wandb logging
            report_to="wandb",
            run_name="llama3-8b-dpo",
            logging_dir="./logs",
        )

        # Initialize trainer context
        trainer = trl.DPOTrainer(
            model=self.learner.model,
            train_dataset=combined_dataset,
            eval_dataset=playpen_dataset["test"],
            args=config,
            processing_class=self.learner.tokenizer,
            peft_config=LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules="all-linear",
                modules_to_save=["lm_head", "embed_token"],
                task_type="CAUSAL_LM",
            ),
        )

        # Train on the dataset; this will save only the adapters to the checkpoints directory
        trainer.train()

        # Optional: Uncomment these lines to merge and save directly
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(f"models/dpo+lora/llama3-8b")
