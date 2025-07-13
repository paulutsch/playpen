from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
from peft import LoraConfig
from datasets import load_dataset, concatenate_datasets, ClassLabel

from playpen import BasePlayPen

import os
import json
import ast

# For wandb api-key
with open("key.json", "r") as f:
    keys = json.load(f)

os.environ["WANDB_API_KEY"] = keys["wandb"]["api_key"]
os.environ["WANDB_PROJECT"] = "llama3-sft"

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
        # Only use the episodes we are interested to train on: here all episodes with successful outcome
        playpen_dataset = playpen_dataset.filter(lambda episode: episode["meta"]["outcome"] == "success") 
        # We shuffle and split the remaining filtered samples to receive a test split
        playpen_dataset = playpen_dataset.train_test_split(0.2, shuffle=True, seed=42)

        ### SFT-Final-Dataset
        sft_final_dataset = load_dataset("clembench-playpen/SFT-Final-Dataset", split="train")

        def parse_and_clean_sft_messages(example):
            """
            Parses the 'chat' field and cleans messages to ONLY include
            'role' and 'content', explicitly excluding 'type' or any other fields.
            """
            chat_data = []
            try:
                chat_data = json.loads(example["chat"])
            except (json.JSONDecodeError, TypeError):
                try:
                    chat_data = ast.literal_eval(example["chat"])
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse chat field with json or ast for example: {example.get('game_id')}, Error: {e}")
                    return {"messages": []}
            except Exception as e:
                print(f"Unhandled error during chat parsing: {e} for example: {example.get('game_id')}")
                return {"messages": []}

            cleaned_messages = []
            for msg in chat_data:
                current_message = {}
                if "role" in msg and msg["role"] is not None:
                    current_message["role"] = str(msg["role"])
                if "content" in msg and msg["content"] is not None:
                    current_message["content"] = str(msg["content"])
                if "role" in current_message and "content" in current_message:
                    cleaned_messages.append(current_message)
            return {"messages": cleaned_messages}

        sft_final_dataset = sft_final_dataset.map(
            parse_and_clean_sft_messages,
            load_from_cache_file=False,
            remove_columns=["chat"] + [col for col in sft_final_dataset.column_names if col == "messages"]
        )
        
        # TUELU Dataset
        tulu_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")
        # Calculate how many tulu examples to use to keep ratio with combined playpen + sft_final
        total_train_size = len(playpen_dataset["train"]) + len(sft_final_dataset)
        tulu_sub_ratio = total_train_size / len(tulu_dataset)
    
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

        assert len(tulu_sub_dataset) == total_train_size, \
            f"Length mismatch: tulu_sub_dataset={len(tulu_sub_dataset)}, combined playpen+sft_final={total_train_size}"

        combined_dataset = concatenate_datasets([playpen_dataset["train"], sft_final_dataset, tulu_sub_dataset])
        print(f"Size of the combined train set: {len(combined_dataset)}")

        # Initialize training configuration
        config = trl.SFTConfig(  # inherits TrainingArguments
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
            #output_dir=f"models/sft+lora/{self.learner.get_name()}",
            output_dir=f"models/sft+lora/llama3-8b",
            eval_strategy="epoch",
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            report_to="wandb",  
            run_name="llama3-8b-sft",
            logging_dir="./logs",
        )

        # Initialize trainer context
        trainer = trl.SFTTrainer(
            model=self.learner.model,
            train_dataset=combined_dataset,
            eval_dataset=playpen_dataset["test"],
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
        #trainer.train(resume_from_checkpoint="models/sft+lora/llama3-8b/checkpoint-400")
        trainer.train()

        # Optional: Uncomment these lines to merge and save directly
        merged_model = trainer.model.merge_and_unload()
        #merged_model.save_pretrained(f"models/sft+lora/{self.learner.get_name()}")
        merged_model.save_pretrained(f"models/sft+lora/llama3-8b")
    
