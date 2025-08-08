import ast
import json
import os
import random
import sys

import torch
import trl
from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry
from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig, DataCollatorWithPadding, GenerationConfig

from examples.trl.ppo_trainer_lora.ppo_trainer import DataCollatorWithGame, PPOTrainer
from examples.trl.ppo_trainer_lora.scorers import DefaultScorer
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
        ppo_dataset = load_dataset(
            "parquet",
            data_files="examples/trl/ppo_trainer_lora/data.parquet",
            split="train",
        )
        ppo_dataset_train, ppo_dataset_eval = ppo_dataset.train_test_split(
            test_size=0.1
        )

        # like the value model, this is a minimal reward model compatible with TRL's get_reward()
        class RewardModel(torch.nn.Module):
            """
            Minimal reward model compatible with TRL's get_reward().
            It must:
              • expose `base_model_prefix`
              • hold an attribute with that name that is a transformer backbone
              • implement `.score(hidden_states)` returning (batch, seq_len) or (batch,) rewards
            """

            base_model_prefix = "model"

            def __init__(self, backbone, tokenizer):
                super().__init__()
                # share the backbone of the policy to derive the output later — yes, it's inconvenient and so is life
                self.model = backbone
                self.tokenizer = tokenizer
                hidden_size = backbone.config.hidden_size

                # match dtype & device to backbone parameters
                ref_param = next(backbone.parameters())
                # self.scorers = {
                #     "wordle": WordleScorer(),
                #     # …
                # }
                self.scorer = DefaultScorer()
                hidden_size = backbone.config.hidden_size
                ref = next(backbone.parameters())
                self.reward_head = torch.nn.Linear(
                    hidden_size, 1, bias=False, device=ref.device, dtype=ref.dtype
                )

            def score(self, hidden_states, games, input_ids, **kwargs):
                """Return (B, T) reward tensor broadcast per response token."""
                texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                batch_rewards = []

                with torch.no_grad():
                    logits = self.model.lm_head(
                        hidden_states
                    )  # (batch, seq_len, vocab)
                    token_ids = logits.argmax(-1)  # TODO: sample?
                    texts = self.tokenizer.batch_decode(
                        token_ids,
                        # skip_special_tokens=True
                    )

                for txt, game in zip(texts, games):
                    prefix_messages, response = txt.rsplit("<|assistant|>", 1)

                    # scorer = self.scorers[game]
                    scorer = self.scorer
                    reward_scalar = scorer.score_turn(prefix_messages, response)
                    batch_rewards.append(reward_scalar)

                # (B,)  →  (B, 1)  →  (B, seq_len) to keep PPOTrainer happy
                reward_vec = torch.tensor(batch_rewards, device=hidden_states.device)
                reward_vec = reward_vec.unsqueeze(1).expand_as(hidden_states[..., 0])
                return reward_vec

        reward_model = RewardModel(self.learner.model, self.learner.tokenizer)

        # based on trl's ppo_trainer.py and utils.py — let's see if at least the architecture works
        class ValueModel(torch.nn.Module):
            """Wraps the policy backbone and projects hidden states to scalar values."""

            base_model_prefix = "model"  # so PPOTrainer can locate the backbone

            def __init__(self, backbone):
                super().__init__()
                self.model = backbone
                hidden_size = backbone.config.hidden_size

                # match dtype & device of the backbone
                ref_param = next(backbone.parameters())
                self.value_head = torch.nn.Linear(
                    hidden_size,
                    1,
                    bias=False,
                    device=ref_param.device,
                    dtype=ref_param.dtype,
                )

            def score(self, hidden_states, **kwargs):
                # hidden_states: (batch, seq_len, hidden_size)
                return self.value_head(
                    hidden_states.to(self.value_head.weight.dtype)
                ).squeeze(
                    -1
                )  # (batch, seq_len)

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                # running backbone with hidden-state outputs in returned.hidden_states
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    **kwargs,
                )

        value_model = ValueModel(self.learner.model)

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

        # i guess PPO expects tokenised prompts with `input_ids`?!
        # see: https://huggingface.co/docs/trl/v0.7.4/en/ppo_trainer
        def tokenize(sample):
            prefix = sample["query"]
            game = sample["game"]
            tok = self.learner.tokenizer(
                prefix,
                truncation=True,
                padding=False,
                return_attention_mask=True,
            )
            return {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "game": game,
            }

        ppo_dataset_train = ppo_dataset_train.map(tokenize, batched=False)
        ppo_dataset_eval = ppo_dataset_eval.map(tokenize, batched=False)

        cols_to_drop = [
            col
            for col in ppo_dataset_train.column_names
            if col not in ("input_ids", "attention_mask", "game")
        ]
        if cols_to_drop:
            ppo_dataset_train = ppo_dataset_train.remove_columns(cols_to_drop)
            ppo_dataset_eval = ppo_dataset_eval.remove_columns(cols_to_drop)

        # policy must expose `.generation_config`?! (missing in TRL ≤ 0.12 wrappers)
        if not hasattr(self.learner.model, "generation_config"):
            self.learner.model.generation_config = GenerationConfig.from_pretrained(
                self.learner.model.config._name_or_path
            )

        trainer = PPOTrainer(
            model=self.learner.model,
            ref_model=trl.create_reference_model(self.learner.model),
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=ppo_dataset_train,
            eval_dataset=ppo_dataset_eval,
            args=config,
            data_collator=DataCollatorWithGame(self.learner.tokenizer),
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

        trainer.train()

        # save the underlying model (model.policy)
        merged_model = trainer.model.policy.merge_and_unload()
        merged_model.save_pretrained("models/ppo+lora/llama3-8b")
