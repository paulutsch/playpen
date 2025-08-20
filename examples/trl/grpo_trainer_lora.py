import json
import os
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import trl
from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry
from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig

from grpo_rewards import (
    reward_imagegame,
    reward_referencegame,
    reward_taboo,
    reward_wordle_withcritic,
)
from playpen import BasePlayPen

# For wandb api-key
with open("key.json", "r") as f:
    keys = json.load(f)

os.environ["WANDB_API_KEY"] = keys["wandb"]["api_key"]
os.environ["WANDB_PROJECT"] = "llama3-dpo"


def calculate_reward(completions, **kwargs):
    """
    Main reward function that routes to the appropriate game-specific reward function
    based on the 'game' column in the dataset.

    Args:
        completions: List of completions to score
        **kwargs: Additional arguments

    Returns:
        List of float rewards in range [-1, 1]
    """
    print("--------------- KWARGS-----------------")
    print(", ".join(k for k, _ in kwargs.items()))

    games = kwargs["game"]
    prompts = kwargs["prompts"]

    scores = []
    for i, (completion, game, prompt) in enumerate(zip(completions, games, prompts)):
        # I double and triple checked these, and they are correct
        # print("====== GAME, COMPLETION, first and last message in prefix ===========")
        # print("------------- GAME -------------------")
        # print(game)
        # print("------------- COMPLETION -------------------")
        # print(completion)
        # print("------------- FIRST MESSAGE IN PREFIX -------------------")
        # print(prompt[0])
        # print("------------- LAST MESSAGE IN PREFIX -------------------")
        # print(prompt[-1])

        completion = completion[0]["content"]

        if game == "imagegame":
            score = reward_imagegame(completion, prompt)
        elif game == "referencegame":
            score = reward_referencegame(completion, prompt)
        elif game == "taboo":
            score = reward_taboo(completion, prompt)
        # elif game == "wordle":
        #     score = reward_wordle(completion, prompt)
        elif game == "wordle_withcritic":
            score = reward_wordle_withcritic(completion, prompt)
        else:
            raise ValueError(f"Unknown game '{game}'")

        scores.append(score)

    return scores


class PeftGrpoTrainer(BasePlayPen):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
        # Note: We configure the proper chat template for the tokenizer already during model loading in the backend

    def learn(self, game_registry: GameRegistry):
        # --- Dataset loading ---

        playpen_dataset = load_dataset("clembench-playpen/DPO_turn", split="train")
        playpen_dataset = playpen_dataset.shuffle(seed=42)

        # wordle and wordle with clue are broken, so we exclude them (check the samples, you'll see)
        def _exclude_broken_games(example):
            return example.get("game") not in {"wordle", "wordle_withclue"}

        playpen_dataset = playpen_dataset.filter(_exclude_broken_games)

        playpen_dataset = playpen_dataset.train_test_split(0.2, shuffle=True, seed=42)

        # assuming sample to have a "prompt" field in the form of [{"role": "user", "content": "..."}, ...]
        def make_conversation(sample):
            new_sample = {
                "prompt": [
                    *sample["prompt"],
                ],
            }
            # drop all other fields
            new_sample.update({k: v for k, v in sample.items() if k not in new_sample})
            return new_sample

        playpen_dataset = playpen_dataset.map(make_conversation)

        print(playpen_dataset["train"][0])

        # Initialize training configuration for grpo
        config = trl.GRPOConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=5e-6,
            num_train_epochs=2,
            lr_scheduler_type="linear",
            warmup_ratio=0.03,
            bf16=True,
            gradient_checkpointing=True,
            seed=7331,
            output_dir=f"models/grpo+lora/llama3-8b",
            logging_steps=10,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            # wandb logging
            report_to="wandb",
            run_name="llama3-8b-grpo",
            logging_dir="./logs",
        )

        # Initialize trainer context
        trainer = trl.GRPOTrainer(
            model=self.learner.model,
            train_dataset=playpen_dataset["train"],
            eval_dataset=playpen_dataset["test"],
            reward_funcs=calculate_reward,
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
        merged_model.save_pretrained(f"models/grpo+lora/llama3-8b")
