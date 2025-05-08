# Playpen

All you need to get started with the LM Playpen Environment for Learning in Interaction.

### Set up the playpen workspace

Clone the repository and switch into the workspace.
```bash
git clone https://github.com/lm-playpen/playpen.git && cd playpen
```

Set up the Python environment. Note: Playpen requires Python 3.10+.
```bash
python -m venv venv --system-site-packages && source venv/bin/activate
```

Install the clemcore framework to run games, backends and models. 
Note: The `hugginface` extra is (only) necessary to train with local hf models.
```bash
pip install clemcore[huggingface]==2.2.1
```

Make playpen available via CLI and install TRL enable running the examples.
```bash
pip install '.[trl]'
```

Make the clembench games available for learning e.g. taboo.
```bash
git clone https://github.com/clp-research/clembench
```

### Dry-run the TRL examples
Dry-run the PPO TRL trainer example with a mock learner (`-l`) and mock teacher (`-t`).
This creates a _playpen-records_ directory containing the mock interactions
```bash
playpen examples/trl/ppo_trainer.py -l mock -t mock
```

### Running the TRL examples with local HF models

Run the PPO TRL trainer example with a Llama3-8b learner (`-l`) and
Llama3-8b teacher (`-t`) model (for 2-player games) using max token length (`-L`) 300.
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