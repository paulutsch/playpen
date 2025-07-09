import trl
from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry
from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig

from playpen import BasePlayPen


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

        # need to convert playpen dataset to match tulu format
        def convert_tulu_to_playpen_format(example):
            example["prompt"] = [{"role": "user", "content": example["prompt"]}]

            return example

        tulu_sub_dataset = tulu_sub_dataset.map(convert_tulu_to_playpen_format)

        print("=== PLAYPEN DATASET FIRST EXAMPLE ===")
        print(playpen_dataset["train"][0])
        print("\n=== TULU DATASET FIRST EXAMPLE ===")
        print(tulu_sub_dataset[0])

        assert len(tulu_sub_dataset) == len(
            playpen_dataset["train"]
        ), f"Length mismatch: tulu_sub_dataset={len(tulu_sub_dataset)}, clembench dataset={len(playpen_dataset['train'])}"

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
            eval_strategy="epoch",
            logging_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            # dpo-specific params
            beta=0.1,  # temperature for dpo loss — Tülu uses beta=5, but they use length-normalized DPO instead of a sum-level loss
        )

        # Initialize trainer context
        trainer = trl.DPOTrainer(
            model=self.learner.model,
            ref_model=None,  # Will use the same model as reference
            train_dataset=combined_dataset,
            eval_dataset=playpen_dataset["test"],
            args=config,
            processing_class=self.learner.tokenizer,
            # see https://huggingface.co/docs/trl/dpo_trainer#training-adapters
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
