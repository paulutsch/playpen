import ast
import json
import os

import torch
import trl
from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry
from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig

from playpen import BasePlayPen

# For wandb api-key
with open("key.json", "r") as f:
    keys = json.load(f)

os.environ["WANDB_API_KEY"] = keys["wandb"]["api_key"]
os.environ["WANDB_PROJECT"] = "llama3-ppo"


class PeftPpoTrainer(BasePlayPen):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        # Note: We configure the proper chat template for the tokenizer already during model loading in the backend

    def learn(self, game_registry: GameRegistry):
        ### TODO: create PPO data
        # sft_final_dataset = load_dataset("clembench-playpen/SFT-Final-Dataset", split="train")

        # this is just a mock! not used during training.
        class RewardModel(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                scores = [1.0 for t in texts]
                return torch.tensor(scores, device=input_ids.device)

        reward_model = RewardModel()

        # Initialize training configuration
        config = trl.PPOConfig(
            per_device_train_batch_size=2,
            seed=7331,
            gradient_accumulation_steps=8,
            warmup_ratio=0.03,
            gradient_checkpointing=True,
            bf16=True,
            learning_rate=5e-6,
            lr_scheduler_type="linear",
            mini_batch_size=5,
            batch_size=20,
            output_dir=f"models/ppo+lora/llama3-8b",
            report_to="wandb",
            run_name="llama3-8b-ppo",
            logging_dir="./logs",
            eval_strategy="epoch",
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
        )

        mock_prompts = [
            "Tell me a joke about entropy.",
            "How do I steal socks from laundromats?",
        ]
        mock_dataset = Dataset.from_dict({"query": mock_prompts})

        # Initialize trainer context
        trainer = trl.PPOTrainer(
            model=self.learner.model,
            ref_model=trl.create_reference_model(self.learner.model),
            reward_model=reward_model,  # this will not be used — we have our own loop, kind of bypassing trl (see below)
            train_dataset=mock_dataset,  # this will not be used — we have our own loop, kind of bypassing trl (see below)
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

        # train model for one step with ppo
        # see: https://medium.com/@chnwsw01/rlhf-with-trl-ppotrainer-6567f3e073a5
        # TODO: make the loop "batcheable"
        for i in range(1):
            # encode a query (this will be the game prompt, taken from the dataset)
            query_txt = "This morning I went to the "
            query_tensor = self.learner.tokenizer.encode(query_txt, return_tensors="pt")

            # get model response (replaced the deprecated respond_to_batch with PPOTrainer.generate)
            # see: https://github.com/huggingface/trl/issues/698
            response_tensor = trainer.generate(self.learner.model, query_tensor)

            # TODO: define a reward for response
            # in here, we check the player response to get the reward (we'll have to access the GameMaster somehow)
            reward = [torch.tensor(1.0)]

            train_stats = trainer.step([query_tensor[0]], [response_tensor[0]], reward)

            print(f"train_stats: {train_stats}")
            print(f"query_tensor: {query_tensor}")
            print(f"query_tensor[0]: {query_tensor[0]}")
            print(f"response_tensor[0]: {response_tensor[0]}")

        # Optional: Uncomment these lines to merge and save directly
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(f"models/ppo+lora/llama3-8b")
