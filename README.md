# Environment

We recommend using a Conda environment to run the Python scripts for this project. Follow these commands to set up the environment and install the required libraries:

```bash
conda create -n abba python=3.10
conda activate abba
pip install -r requirements.txt
```

# Arithmetic Reasoning

To train the models, execute:

```bash
bash scripts/train_arithmetic.sh
```

This script will fine-tune a model on the MetaMathQA dataset. You can modify the `model` parameter to use a different model if desired. The script will save the fine-tuned adapters.

Run the following to evaluate on GSM8K and MATH benchmarks:
```bash
bash scripts/arithmetic_merge_eval.sh
```

# Commonsense Reasoning

To run the commonsense experiments, start by downloading the required datasets.

Begin by fetching the fine-tuning dataset available [here](XXXX). Place this file in the `data/commonsense` folder.

Next, for the evaluation phase, download the necessary datasets from [this link](XXXX). Ensure each dataset is saved in its appropriate subdirectory within `data/commonsense`.

To train the models, use:

```bash
bash scripts/train_cr.sh
```

This script will fine-tune a model on the Commonsense170K dataset. You can modify the `model` parameter to explore various models. The script will save the fine-tuned adapters.

Run the following to evaluate on commonsense reasoning benchmarks:
```bash
bash scripts/cr_merge_eval.sh
```