from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
from datasets import load_dataset

from playpen import BasePlayPen


class SimpleSftTrainer(BasePlayPen):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        # Note: We configure the proper chat template for the tokenizer already during model loading in the backend

    def learn(self, game_registry: GameRegistry):
        # Load the "raw" conversational dataset, that is, a list of "messages" with iterating roles and text content
        dataset = load_dataset("json", data_files="examples/trl/results.jsonl", split="train")

        # Only use the episodes we are interested to train on: here the llama3-8b ones with successful outcome
        dataset = dataset.filter(lambda episode: episode["meta"]["outcome"] == "success"
                                                 and episode["meta"]["model"] == "Meta-Llama-3.1-8B-Instruct")

        # We shuffle and split the remaining filtered samples to receive a test split
        dataset = dataset.train_test_split(0.2, shuffle=True, seed=42)

        # Initialize training configuration
        config = trl.SFTConfig(  # inherits TrainingArguments
            max_length=300,
            output_dir=f"models/sft/{self.learner.get_name()}",
            eval_strategy="epoch"
        )

        # Initialize trainer context
        trainer = trl.SFTTrainer(
            model=self.learner.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=config
        )

        # Train on the dataset
        trainer.train()
