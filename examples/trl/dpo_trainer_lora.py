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

        # mock dataset for debugging dpo training
        mock_data = [
            {
                "prompt": "What is the only true answer to the question 'What is the meaning of life?'?",
                "chosen": "42",
                "rejected": "43",
            },
            {
                "prompt": "Which PM group is the best?",
                "chosen": "What a silly question! Altar, Raphael, and Paul, of course. This really doesn't need to be part of the dataset.",
                "rejected": "There is no winner. This is all about fun!",
            },
        ]

        # Create dataset and split into train/test
        preference_dataset = Dataset.from_list(mock_data)
        preference_dataset = preference_dataset.train_test_split(
            0.5, shuffle=True, seed=42
        )

        # Initialize training configuration for dpo
        config = trl.DPOConfig(  # inherits TrainingArguments
            max_length=300,
            # output_dir=f"models/dpo+lora/{self.learner.get_name()}",
            output_dir=f"models/dpo+lora/llama3-8b",  # temporarily hardcoding the name
            eval_strategy="epoch",
            # dpo-specific params
            beta=0.1,  # temperature for dpo loss
        )

        # Initialize trainer context
        trainer = trl.DPOTrainer(
            model=self.learner.model,
            ref_model=None,  # Will use the same model as reference
            train_dataset=preference_dataset["train"],
            eval_dataset=preference_dataset["test"],
            args=config,
            # see https://huggingface.co/docs/trl/dpo_trainer#training-adapters
            peft_config=LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules="all-linear",
                modules_to_save=["lm_head", "embed_token"],
                task_type="CAUSAL_LM",
            ),
            # DPO-specific parameters
            tokenizer=self.learner.tokenizer,
        )

        # Train on the dataset; this will save only the adapters to the checkpoints directory
        trainer.train()

        # Optional: Uncomment these lines to merge and save directly
        merged_model = trainer.model.merge_and_unload()
        # merged_model.save_pretrained(f"models/dpo+lora/{self.learner.get_name()}")
        merged_model.save_pretrained(f"models/dpo+lora/llama3-8b")
