#!/bin/bash

echo "starting..."

HF_HOME="$(pwd)/.cache"
export HF_HOME
HF_TOKEN="$(cat .hf_token)"
export HF_TOKEN
WANDB_MODE="offline"
export WANDB_MODE

source activate reft
# run this in interactive job if you want to see GPU util, otherwise it runs forever
# nvidia-smi -l 2 # print GPU util on 2s loop
python reft_demo.py

echo "ending"
