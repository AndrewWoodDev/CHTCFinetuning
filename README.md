# Refined Evaluation and Fine-Tuning Toolkit

This repository provides tools and demonstrations for evaluating and fine-tuning large language models using a refined reward modeling framework.

## Structure

- `reft_demo.py` / `reft_demo.ipynb`: Entry point for demonstrating model reward evaluation and generation.
- `reft_OG/`: Core implementation of reward-based fine-tuning and evaluation modules.
  - `examples/`: Includes multiple subfolders showcasing practical applications including LoRA, ICL, DPO, and more.
  - `pyreft/`: Core configuration, trainer, and model adaptation code.
- `reft_and_lora/`: Simple contrast setup between pure LoRA and integrated Reft+LoRA.

## Key Features

- Modular training and evaluation of models using custom rewards
- LoRA and reward intervention compatibility
- End-to-end demo notebooks for ICL, reward tuning, memorization, safety, and more
- Plotting utilities for evaluating training performance

## Usage

1. Install dependencies listed in your environment or Dockerfile.
2. Run one of the provided demo scripts or notebooks.
3. Customize datasets, templates, or reward functions as needed.

## License

This project is made available for research and educational purposes.

## Setting up the environment:
```
conda create -n reft python=3.10 # python >=3.9 is required
conda activate reft
# IMPORTANT: check your CUDA version before running the following step and match w/ that instead
conda install transformers pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia #I believe it is 12.4 on the lab machines
pip install nnsight pyreft
```

## Running on CHTC:
1. Make a `.hf_token` and put your HuggingFace token directly in there
2. Make sure you have access to LLaMa 2
3. Run the standard `condor_submit train.sub` on a CHTC machine
4. You can also run the Docker container in interactive mode using the `-i` flag on previous command. Note that it won't execute the `exec.sh` though--you'll have to do that yourself if you want it to run

