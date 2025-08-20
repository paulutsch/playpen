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
        self.has_shown_example = False

        ppo_dataset = load_dataset(
            "parquet",
            data_files="examples/trl/ppo_trainer_lora/data.parquet",
            split="train",
        )

        splits = ppo_dataset.train_test_split(test_size=0.1, seed=42)
        ppo_dataset_train = splits["train"].select(range(100))
        ppo_dataset_eval = splits["test"].select(range(100))

        print("example data point", ppo_dataset_train[0])

        # Debug: Check dataset structure
        print("=== Dataset Structure Debug ===")
        print("Dataset columns:", ppo_dataset_train.column_names)
        print("First example keys:", list(ppo_dataset_train[0].keys()))
        for key, value in ppo_dataset_train[0].items():
            print(f"  {key}: {type(value)} = {value}")
        print("=== End Dataset Structure ===")

        class RewardModel(torch.nn.Module):
            """
            Minimal reward model compatible with TRL's get_reward() *without*
            trying to move the full 8‑B‑param backbone a second time.
            We keep a *reference* to the learner's backbone but do NOT register
            it as a sub‑module.  Only the tiny reward_head is an actual parameter.
            """

            # utils.get_reward() looks for this attr
            base_model_prefix = "_backbone"

            def __init__(self, backbone, tokenizer):
                super().__init__()
                # keep unregistered pointer to the big model (avoid relocation via .to())
                object.__setattr__(self, "_backbone", backbone)

                self.tokenizer = tokenizer

                self.scorer = DefaultScorer()

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                return self._backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    **kwargs,
                )

            def score(self, hidden_states, *, games, input_ids, context_length, **_):
                # texts = self.tokenizer.batch_decode(
                #     input_ids, skip_special_tokens=False
                # )
                batch_rewards = []

                # for txt in texts:
                #     prefix_messages, response = txt.rsplit(
                #         "<|start_header_id|>assistant<|end_header_id|>\n\n", 1
                #     )
                #     print("=== reward score prefix ===\n", prefix_messages)
                #     print("=== reward score response ===\n", response)
                #     batch_rewards.append(
                #         self.scorer.score_turn(prefix_messages, response)
                #     )

                # Handle context_length as either int or list
                if isinstance(context_length, int):
                    # Single context length for all sequences
                    for ids, game in zip(input_ids, games):
                        q_ids = ids[:context_length]
                        r_ids = ids[context_length:]
                        prefix_messages = self.tokenizer.decode(
                            q_ids, skip_special_tokens=False
                        )
                        response = self.tokenizer.decode(
                            r_ids, skip_special_tokens=False
                        )
                        print(
                            "=== QUERY ===\n",
                            prefix_messages,
                            "\n=== RESP ===\n",
                            response,
                        )
                        batch_rewards.append(
                            self.scorer.score_turn(prefix_messages, response)
                        )
                else:
                    # List of context lengths - process each sequence individually
                    for ids, game, ctx_len in zip(input_ids, games, context_length):
                        q_ids = ids[:ctx_len]
                        r_ids = ids[ctx_len:]
                        prefix_messages = self.tokenizer.decode(
                            q_ids, skip_special_tokens=False
                        )
                        response = self.tokenizer.decode(
                            r_ids, skip_special_tokens=False
                        )
                        print(
                            "=== QUERY ===\n",
                            prefix_messages,
                            "\n=== RESP ===\n",
                            response,
                        )
                        batch_rewards.append(
                            self.scorer.score_turn(prefix_messages, response)
                        )

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

            def score(self, hidden_states, *, input_ids, **_):
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
            output_dir=f"models/ppo+lora/llama3-8b",
            report_to="wandb",
            run_name="llama3-8b-ppo",
            logging_dir="./logs",
            eval_strategy="epoch",
            logging_steps=1,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
        )

        # policy must expose `.generation_config`?! (missing in TRL ≤ 0.12 wrappers)
        if not hasattr(self.learner.model, "generation_config"):
            self.learner.model.generation_config = GenerationConfig.from_pretrained(
                self.learner.model.config._name_or_path
            )

        # Fix: Use proper pad token instead of custom string
        if self.learner.tokenizer.pad_token is None:
            # Try to use the model's native pad token, fallback to EOS
            if (
                hasattr(self.learner.tokenizer, "pad_token")
                and self.learner.tokenizer.pad_token is not None
            ):
                print("=== PAD token already set ===")
                pass  # Already set
            elif (
                hasattr(self.learner.tokenizer, "eos_token")
                and self.learner.tokenizer.eos_token is not None
            ):
                print("=== EOS token found, using it as pad token ===")
                self.learner.tokenizer.pad_token = self.learner.tokenizer.eos_token
            else:
                print("=== No pad token found, using </s> as pad token ===")
                # Last resort: use a common pad token
                self.learner.tokenizer.pad_token = "</s>"

        # Ensure pad_token_id is set
        if self.learner.tokenizer.pad_token_id is None:
            self.learner.tokenizer.pad_token_id = self.learner.tokenizer.eos_token_id

        # Debug: Check tokenizer special tokens
        print("=== Tokenizer Debug Info ===")
        print(
            f"EOS token: {self.learner.tokenizer.eos_token} (ID: {self.learner.tokenizer.eos_token_id})"
        )
        print(
            f"PAD token: {self.learner.tokenizer.pad_token} (ID: {self.learner.tokenizer.pad_token_id})"
        )
        print(
            f"BOS token: {self.learner.tokenizer.bos_token} (ID: {self.learner.tokenizer.bos_token_id})"
        )

        # Check if EOT token exists in vocabulary
        eot_token = "<|eot_id|>"
        if eot_token in self.learner.tokenizer.get_vocab():
            eot_id = self.learner.tokenizer.get_vocab()[eot_token]
            print(f"EOT token: {eot_token} (ID: {eot_id})")
        else:
            print(f"EOT token '{eot_token}' NOT found in vocabulary!")

        # Test tokenization of EOT token
        test_tokens = self.learner.tokenizer.encode(eot_token, add_special_tokens=False)
        test_decode = self.learner.tokenizer.decode(
            test_tokens, skip_special_tokens=False
        )
        print(
            f"EOT tokenization test: '{eot_token}' -> {test_tokens} -> '{test_decode}'"
        )

        # Test tokenization of the problematic string
        test_pad = self.learner.tokenizer.encode("<|pad_id|>", add_special_tokens=False)
        test_pad_decode = self.learner.tokenizer.decode(
            test_pad, skip_special_tokens=False
        )
        print(f"Pad token test: '<|pad_id|>' -> {test_pad} -> '{test_pad_decode}'")

        # Check if EOT token is problematic and needs replacement
        eot_token = "<|eot_id|>"
        if eot_token in self.learner.tokenizer.get_vocab():
            eot_id = self.learner.tokenizer.get_vocab()[eot_token]
            test_eot = self.learner.tokenizer.encode(
                eot_token, add_special_tokens=False
            )
            test_eot_decode = self.learner.tokenizer.decode(
                test_eot, skip_special_tokens=False
            )
            print(f"EOT token test: '{eot_token}' -> {test_eot} -> '{test_eot_decode}'")

            # If EOT token decodes to something unexpected, we might need to replace it
            if test_eot_decode != eot_token:
                print(
                    f"WARNING: EOT token decodes to '{test_eot_decode}' instead of '{eot_token}'"
                )
                print(
                    "This might cause issues during training. Consider replacing EOT with EOS in your data."
                )

        print("=== End Tokenizer Debug ===")

        # i guess PPO expects tokenised prompts with `input_ids`?!
        # see: https://huggingface.co/docs/trl/v0.7.4/en/ppo_trainer
        def tokenize(sample):
            prefix = sample["query"]
            prefix, _ = prefix.rsplit(
                "<|start_header_id|>assistant<|end_header_id|>", 1
            )
            prefix = prefix + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            game = sample["game"]

            # Tokenize without padding first to get true context length
            tok = self.learner.tokenizer(
                prefix,
                add_special_tokens=False,  # already part of the string
                truncation=True,
                padding=False,
                return_attention_mask=True,
            )

            # Store the original context length before any padding
            context_length = len(tok["input_ids"])

            text = self.learner.tokenizer.decode(
                tok["input_ids"], skip_special_tokens=False
            )
            if not self.has_shown_example:
                print("=== prefix ===", prefix)
                print("=== text ===", text)
                print("=== context_length ===", context_length)
                print("=== input_ids ===", tok["input_ids"])

                # Check if EOT token appears in the tokenized sequence
                eot_id = self.learner.tokenizer.get_vocab().get("<|eot_id|>")
                if eot_id is not None:
                    eot_positions = [
                        i
                        for i, token_id in enumerate(tok["input_ids"])
                        if token_id == eot_id
                    ]
                    print(f"=== EOT token positions: {eot_positions}")

                # Test what happens when we decode individual tokens
                for i, token_id in enumerate(tok["input_ids"][:10]):  # First 10 tokens
                    decoded = self.learner.tokenizer.decode(
                        [token_id], skip_special_tokens=False
                    )
                    print(f"Token {i}: {token_id} -> '{decoded}'")

                self.has_shown_example = True
            return {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "game": game,
                "context_length": context_length,  # Store original context length
            }

        ppo_dataset_train = ppo_dataset_train.map(tokenize, batched=False)
        ppo_dataset_eval = ppo_dataset_eval.map(tokenize, batched=False)
        print(
            "===! ppo_dataset_train ===",
            self.learner.tokenizer.decode(
                ppo_dataset_train[0]["input_ids"], skip_special_tokens=False
            ),
        )

        # Debug: Check what fields remain after tokenization
        print("=== After Tokenization Debug ===")
        print("Dataset columns after tokenization:", ppo_dataset_train.column_names)
        print("First example after tokenization:")
        for key, value in ppo_dataset_train[0].items():
            print(f"  {key}: {type(value)} = {value}")
        print("=== End After Tokenization ===")

        # Explicitly keep only the columns we need
        cols_to_keep = ("input_ids", "attention_mask", "game", "context_length")
        cols_to_drop = [
            col for col in ppo_dataset_train.column_names if col not in cols_to_keep
        ]
        print(f"=== Dropping columns: {cols_to_drop} ===")
        if cols_to_drop:
            ppo_dataset_train = ppo_dataset_train.remove_columns(cols_to_drop)
            ppo_dataset_eval = ppo_dataset_eval.remove_columns(cols_to_drop)

        print(f"=== Final columns: {ppo_dataset_train.column_names} ===")

        # Debug: Check what happens after data collation
        print("=== Testing Data Collation ===")
        test_batch = [ppo_dataset_train[0], ppo_dataset_train[1]]
        collator = DataCollatorWithGame(self.learner.tokenizer)
        collated = collator(test_batch)

        print("Original token 0:", ppo_dataset_train[0]["input_ids"])
        print("Collated token 0:", collated["input_ids"][0])
        print(
            "Original decode 0:",
            self.learner.tokenizer.decode(
                ppo_dataset_train[0]["input_ids"], skip_special_tokens=False
            ),
        )
        print(
            "Collated decode 0:",
            self.learner.tokenizer.decode(
                collated["input_ids"][0], skip_special_tokens=False
            ),
        )

        # Check for EOT token in collated data
        eot_id = self.learner.tokenizer.get_vocab().get("<|eot_id|>")
        if eot_id is not None:
            eot_positions = [
                i
                for i, token_id in enumerate(collated["input_ids"][0])
                if token_id == eot_id
            ]
            print(f"EOT positions in collated: {eot_positions}")

        print("=== End Data Collation Test ===")

        self.has_shown_example = False

        trainer = PPOTrainer(
            model=self.learner.model,
            ref_model=None,
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
