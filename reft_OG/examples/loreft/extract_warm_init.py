import argparse
from baukit import TraceDict
import torch
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
)
from tqdm import tqdm
import numpy as np
import os

def get_single_activation(model, query):
    MLPS_OUT = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    input_ids = tokenizer(query, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        with TraceDict(model, MLPS_OUT) as ret:
            output = model(input_ids, output_hidden_states = True)
        mlp_out = [ret[mlp_].output.squeeze().detach().cpu() for mlp_ in MLPS_OUT]
        mlp_out = torch.stack(mlp_out, dim = 0).squeeze().detach()
    return mlp_out[:, -1, :]

def extract_representation():
    true_out_all = []
    counterfactual_out_all = []
    random_sample = np.random.choice(np.array(train_data_all), n_random_sample)
    for data_item in tqdm(random_sample):
        true_ans = data_item["answer"] 
        counterfactual_ans = "true" if true_ans == "false" else "false"
        base_prompt = task_prompt_template % (data_item['instruction'])
        true_input = base_prompt + trigger_tokens + true_ans + tokenizer.eos_token
        counterfactual_input = base_prompt + trigger_tokens + counterfactual_ans + tokenizer.eos_token

        true_embedding = get_single_activation(model, true_input)
        counterfactual_embedding = get_single_activation(model, counterfactual_input)
        
        true_out_all.append(true_embedding)
        counterfactual_out_all.append(counterfactual_embedding)
    true_out_all = torch.stack(true_out_all)
    counterfactual_out_all = torch.stack(counterfactual_out_all)
    torch.save(true_out_all, f"warm_init/commonsense_true_{n_random_sample}.pt")
    torch.save(counterfactual_out_all, f"warm_init/commonsense_counterfactual_{n_random_sample}.pt")

def extract_alignez(pos_emb, neg_emb):
    pos_emb = pos_emb.to(torch.float32)
    neg_emb = neg_emb.to(torch.float32)
    v_pos_all = []
    v_neg_all = []
    for layer_idx in range(32):
        pos_emb_ = pos_emb[:,layer_idx,:].squeeze()
        neg_emb_ = neg_emb[:,layer_idx,:].squeeze()
        _,_,v_pos = torch.svd(pos_emb_, compute_uv=True, some=True)
        _,_,v_neg = torch.svd(neg_emb_, compute_uv=True, some=True)
        v_pos = v_pos.T[0,:]
        v_neg = v_neg.T[0,:]
        v_pos_all.append(v_pos)
        v_neg_all.append(v_neg)
    v_pos_all = torch.vstack(v_pos_all).to(torch.bfloat16)
    v_neg_all = torch.vstack(v_neg_all).to(torch.bfloat16)
    if not os.path.isdir("warm_init/alignez/"):
        os.makedirs("warm_init/alignez/")
    torch.save(v_pos_all, f"warm_init/alignez/v_pos_{n_random_sample}.pt")
    torch.save(v_neg_all, f"warm_init/alignez/v_neg_{n_random_sample}.pt")
    

def extract_iti():
    pass

def extract_caa():
    pass

def extract_repe():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data-path', '--data-path', type=str, required=True)
    parser.add_argument('-method', '--method', type=str, default='alignez')
    parser.add_argument('-model-name', '--model-name', type=str, required=True)
    parser.add_argument('-dtype', '--dtype', type=str, default="bfloat16")
    parser.add_argument('-n', '--n_sample', type=int, default=1000)

    args = parser.parse_args()
    data_path = args.data_path
    method = args.method
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model_name
    dtype = args.dtype
    max_length = 512
    n_random_sample = args.n_sample

    assert method in ['alignez', 'iti', 'caa', 'repe']

    extract_fn_dict = {
        'alignez': extract_alignez,
        'iti': extract_iti,
        'caa': extract_caa,
        'repe': extract_repe,
    }
    extract_fn = extract_fn_dict[method]

    with open(data_path, 'r') as j:
        train_data_all = json.loads(j.read())


    task_prompt_template = "%s\n"
    trigger_tokens = "the correct answer is "
    

    if not os.path.isfile(f'warm_init/embeddings/commonsense_true_{n_random_sample}.pt'):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype if dtype != "float8" else None,  # save memory
            load_in_8bit=True if dtype == "float8" else False,
            device_map=device
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
        )
        extract_representation()
    else:
        pos_emb = torch.load(f'warm_init/embeddings/commonsense_true_{n_random_sample}.pt').to(device)
        neg_emb = torch.load(f'warm_init/embeddings/commonsense_counterfactual_{n_random_sample}.pt').to(device)
        print(pos_emb.shape, neg_emb.shape)
        extract_fn(pos_emb, neg_emb)
    
    

