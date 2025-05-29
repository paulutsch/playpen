![Playpen Logo](./playpen-logo.png)

[Pre-Print](https://arxiv.org/abs/2504.08590)

All you need to get started with the LM Playpen Environment for Learning in Interaction.

## Set up the playpen workspace

Clone the repository and switch into the workspace.
```bash
git clone https://github.com/lm-playpen/playpen.git && cd playpen
```

Set up the Python environment. Note: Playpen requires Python 3.10+.
```bash
python -m venv venv --system-site-packages && source venv/bin/activate
```

Install the clemcore framework to run games, backends and models. 
Note: The `huggingface` extra is (only) necessary to train with local hf models.
```bash
pip install clemcore[huggingface]==2.2.1
```

Make playpen available via CLI and install TRL enable running the examples.
```bash
pip install '.[trl]'
```

Make the clembench games, e.g. taboo, available for learning.
For this, clone the clembench repository to a directory of your choice. 
```bash
git clone https://github.com/clp-research/clembench
```

Furthermore, we must install the clembench game requirements in our venv so that all games can be run properly:   
```bash
pip install your/path/to/clembench/requirements.txt
```


Then, back in you playpen workspace, copy the `game_registry.json.template` to `game_registry.json` so that the `clem` CLI can find it in the current working directory.
Set the path to the directory which contains the clembench repository.
The following command has a similar effect:
```bash
echo '[{"benchmark_path": "your/path/to/clembench"}]' > game_registry.json
```

> **Note:** Adding the game registry file is not necessary, 
> when you clone the clembench repository directly in your playpen workspace. 
> In this case the clem CLI can directly find the games by looking into sub-directories.
 
In any case, check that games are available via:
```bash
clem list games
```

## Supervised Finetuning

Supervised fine-tuning (SFT) is known to help learning in interaction as it shifts the model's distribution towards the interactive data it will operate on.

In the context of clembench this means to let the model observe patterns of interaction which occur in various dialogue games. 

### Determine model performance before training

First, we want to know how well the model we want to fine-tune already performs on the benchmark.
For this we let the model play the games in the benchmark. 


The following command runs the `smol-135m` model on the games that were selected for version `2.0` of the benchmark: 
```bash
clem run -g "{'benchmark':['2.0']}" -m smol-135m 
```

Alternatively, you can also use `-g all` to run all games or, for example, `-g taboo` to run just a specific game.

When the `run` command finished, a `results` folder should appear which contains the recorded interaction in the game.
We score the interactions with the following command:

```bash
clem score # Note: -r option defaults to 'results' 
```

Alternatively, you can also use, for example, `-g taboo` to score just a specific game.

The score command creates `score.json` in each episode directory which we use for the overall evaluation.
The evaluation runs with the following command:

```bash
clem eval # Note: -r option defaults to 'results'
```

This creates a `results.csv` and `results.html` which presents the overall aggregated result and for each game individually. 

> **Note:** The `clem` CLI has become available during workspace setup when we installed `clemcore`. In addition, we checked out the `clembench` repository that contains the games.
> 

> **Note:** We choose `smol-135m` only to showcase the workflow. For real training you should more capable models e.g. `llama3-8b`.
> You can look up baseline performances of other models on the leaderboard: https://clembench.github.io/leaderboard.html.

### Create the conversational dataset for training

The prepared examples make use of huggingface datasets.
Hence, we convert the interactions recorded in the `interactions.json` into a conversational dataset.
The main property of such a dataset is that it contains samples which specify a list of `messages`.
These messages usually iterate on roles, that is, between a `user` and an `assistant`.

> **Note:** We already prepared a small dataset example. You can find it under `examples/trl/results.jsonl`.
> Hence, you can skip the code segments in this section if you do not want to overwrite that file.
> The full dataset for v2.0 is given in results.jsonl.zip.
> The unpacked file is >100MB in size and therefore git-ignored.

We use the interactions already recorded in https://github.com/clembench/clembench-runs.git.
Hence, we clone the repository (to a place outside of the workspace, because the repository is quite large):

```
# This might take a while since the repository is quite large!
git clone https://github.com/clembench/clembench-runs.git
```

These contain results for each version of the benchmark. We are interested in the model behaviors for version 2.0.
Therefore, we run the following command:

```
python3 examples/trl/data_utils.py <path-to>/clembench-runs/v2.0
```

This will create in `examples/trl/results.jsonl` containing all interactions in form of a conversational dataset.
Furthermore, the script adds a `meta` annotation that informs about 
`game`, `experiment`, `game_id`, `player_name`, `game_role`, `model` and `outcome` 
which can be used for filtering the samples in the dataset.

Notably, the dataset contains samples of interaction from both perspectives of the 2-player games. 
For example, for taboo the dataset contains the same episode, once from the perspective of the guesser and 
once from the perspective of the clue giver.

> **Note:** The default implementation of TRL for SFT only trains the model to predict the last `assistant` messages.
> All other messages are handled as a prefix or context for the prediction.

> **Note:** You can also collect your own data samples by simply running the benchmark as described before 
> and then use the output of the `results` directory for you chosen model.

### Running the SFT example with local HF models

Now we are ready to run the simple SFT TRL trainer example with a `smol-135m` learner (`-l`).
Since the interactions were already performed by other models, we do not need a teacher model in this case.
The following commands runs the example training pipeline:
```bash
playpen run examples/trl/sft_trainer_simple.py -l smol-135m 
```

The `playpen` CLI properly loads the huggingface model and runs the trainer code in the specified file. 
When the command finished successfully, then there will be a `models/sft/smol-135m` directory 
containing a checkpoint folder, e.g. `checkpoint-84` with the updated parameters of the model.

> **Note:** Have a look at `examples/trl/sft_trainer_simple.py` for implementation details.

### Evaluate the fine-tuned model

To evaluate the effectiveness of our SFT approach, we run the trained model again on the clembench.
For this, we first register our trained model in our local `model_registry.json` 
by adding an entry that points to the checkpoint folder:
```json
{
  "model_name": "smol-135m-sft",
  "backend": "huggingface_local",
  "huggingface_id": "models/sft/smol-135m/checkpoint-84",
  "release_date": "2024-09-04",
  "open_weight": true,
  "parameters": "135M",
  "languages": ["en"],
  "context_size": "2048",
  "license": {
    "name": "Apache 2.0",
    "url": "https://www.apache.org/licenses/LICENSE-2.0"
  },
  "model_config": {
    "premade_chat_template": true,
    "eos_to_cull": "<\\|im_end\\|>"
  }
}
```

Then we can run the benchmark again, but this time with `-m smol-135m-sft`:  
```bash
clem run -g "{'benchmark':['2.0']}" -m smol-135m-sft
```

### Parameter Efficient Fine-tuning

tbd

## Learning in Interaction (not yet available)

After SFT we can let the model learn more interactively with a teacher policy playing the other role in 2-player games.

### Running the TRL examples with local HF models

Run the PPO TRL trainer example with a Llama3-8b learner (`-l`) and
Llama3-8b teacher (`-t`) model (for 2-player games) using max token length (`-L`) 300.
This creates a _playpen-records_ directory containing the interactions.
```bash
playpen examples/trl/ppo_trainer.py -l llama3-8b -t llama3-8b -L 300
```

### Running the TRL examples with remote API models

Adding credentials to the key.json for remote API access. 
Note: An full tempalte for all supported remote backends is given in `key.json.template`. 
You can also manually insert the required information there and rename the file to `key.json`. 
```bash
echo '{
  "openai": {
    "organisation": "your_organisation",
    "api_key": "your_api_key"
  }
}' > key.json
```

Run the PPO TRL trainer example with a Llama3-8b learner (`-l`) and a GPT-4 teacher (`-t`) model using temperature (`-T`) 0.7.

```bash
playpen examples/trl/ppo_trainer.py -l llama3-8b -t gpt4o-mini -L 300 -T 0.7
```

## Using other existing models

Rename an already specified model or use another model by adding a custom model registry to the workspace. 

Note: The entry with the renamed model is already prepared in the `model_registry.json` of this repository. The following code snippet exemplifies how this can be done.

Lookup existing (packaged) model specs.
```bash
playpen list models -v | grep Meta -A 6
...
Meta-Llama-3.1-8B-Instruct -> huggingface_local (packaged)
ModelSpec: {"model_name":"Meta-Llama-3.1-8B-Instruct" ... 
...
```

Note: You can also look up the packaged model specs in the [clemcore repository](https://github.com/clp-research/clemcore/blob/maintenance/2.x/clemcore/backends/model_registry.json).

Change model name from Meta-Llama-3.1-8B-Instruct to llama3-8b
```bash
echo '[{
  "model_name": "llama3-8b",
  "backend": "huggingface_local",
  "huggingface_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "release_date": "2024-07-23",
  "open_weight": true,
  "parameters": "8B",
  "languages": ["en", "de", "fr", "it", "pt", "hi", "es", "th"],
  "context_size": "128k",
  "license": {
    "name": "Meta",
    "url": "https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE"
  },
  "model_config": {
    "requires_api_key": true,
    "premade_chat_template": true,
    "eos_to_cull": "<\\|eot_id\\|>"
  }
}]' > model_registry.json 
```
The `llama3-8b` becomes available for model selection via the entry in the custom `model_registry.json`. Note that custom entries always precede packaged entries.
```bash
playpen list models | grep llama3
llama3-8b -> huggingface_local (.../playpen/model_registry.json)
```

If you want to make another existing Huggingface model available, then change here the `huggingface_id`, choose an appropriate `model_name` and set other relevant parameters.

## Implement your own playpen trainer

tbd
