from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
from peft import LoraConfig
from datasets import load_dataset, concatenate_datasets, ClassLabel

from playpen import BasePlayPen


class PeftSftTrainer(BasePlayPen):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        # Note: We configure the proper chat template for the tokenizer already during model loading in the backend

    def learn(self, game_registry: GameRegistry):
        # Load a conversational dataset for SFT, that is, a list of "messages" -- basically tuples of role and content.
        # The role can be "user" or "assistant" and typically alternates within the list.
        # During training, everything up to the last assistant message becomes the prefix for prediction.
        # The loss is calculated based on the differences to the last assistant message.
        # Here we load the canonical training split as available in the huggingface playpen-data repository.
        # By default, the dataset is stored in ~/.cache/huggingface/datasets/ on your machine. This might take a while.
        playpen_dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")
        # try this one later too: clembench-playpen/SFT-Final-Dataset

        # Only use the episodes we are interested to train on: here all episodes with successful outcome
        playpen_dataset = playpen_dataset.filter(lambda episode: episode["meta"]["outcome"] == "success")

        # We shuffle and split the remaining filtered samples to receive a dev split
        # For evaluation on the actual games performance use the validation split
        # load_dataset("json", data_files="examples/trl/results.jsonl", split="validation")
        #dataset = dataset.train_test_split(0.2, shuffle=True, seed=42)
        playpen_dataset = playpen_dataset.train_test_split(0.2, shuffle=True, seed=42)

        # adding the tulu
        tulu_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
        tulu_sub_ratio = len(playpen_dataset["train"]) / len(tulu_dataset)

        # Use only the same number of examples from TÜLU as in the clem dataset, while maintaining the original SFT source distribution.
        unique_sources = tulu_dataset.unique("source")
        source_classlabel = ClassLabel(names=unique_sources) # needed for stratified split
        tulu_dataset = tulu_dataset.cast_column("source", source_classlabel)
        
        tulu_split = tulu_dataset.train_test_split(
            train_size=tulu_sub_ratio,
            stratify_by_column="source", 
            seed=8 #tulu3 uses seed 8 for SFT too
            )
        
        tulu_sub_dataset = tulu_split["train"]

        assert len(tulu_sub_dataset) == len(playpen_dataset), \
            f"Length mismatch: tulu_sub_dataset={len(tulu_sub_dataset)}, clembench dataset={len(playpen_dataset["train"])}"

        combined_dataset = concatenate_datasets([playpen_dataset["train"], tulu_sub_dataset])
        print(f"Size of the train set: {len(combined_dataset)}")
        
        # Initialize training configuration
        config = trl.SFTConfig(  # inherits TrainingArguments
            max_length=300,
            output_dir=f"models/sft+lora/{self.learner.get_name()}",
            eval_strategy="epoch"
        )

        # Initialize trainer context
        trainer = trl.SFTTrainer(
            model=self.learner.model,
            train_dataset=combined_dataset,
            eval_dataset=playpen_dataset["test"], # only validate on the playpen data to get clem/stat-score
            args=config,
            # see https://huggingface.co/docs/trl/sft_trainer#training-adapters
            peft_config=LoraConfig(
                r=16, lora_alpha=32,
                lora_dropout=0.05,
                target_modules="all-linear",
                modules_to_save=["lm_head", "embed_token"],
                task_type="CAUSAL_LM",
            )
        )

        # Train on the dataset; this will save only the adapters to the checkpoints directory
        trainer.train()

        # Optional: Uncomment these lines to merge and save directly
        # merged_model = trainer.model.merge_and_unload()
        # merged_model.save_pretrained(f"models/sft+lora/{self.learner.get_name()}")
