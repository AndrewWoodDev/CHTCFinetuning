FROM continuumio/miniconda3:24.4.0-0

# Create the conda environment and install necessary packages
RUN conda create -n reft python=3.10 -y && \
    conda clean -afy && \
    /bin/bash -c "source activate reft && \
    conda install -y nvidia/label/cuda-12.4.0::cuda && \
    conda install -y transformers pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia && \
    pip install accelerate bitsandbytes nnsight pyreft tqdm && \
    pip install git+https://github.com/davidbau/baukit"

# # Set the environment variable for the Hugging Face token
# COPY .hf_token /tmp/.hf_token
# RUN echo "HF_TOKEN=$(cat /tmp/.hf_token)" >> ~/.bashrc
# ENV HF_TOKEN="$(cat /tmp/.hf_token)"
# 
# COPY reft_demo.py reft_demo.py
# 
# CMD ["conda", "activate", "reft", ";", "python", "reft_demo.py"]

