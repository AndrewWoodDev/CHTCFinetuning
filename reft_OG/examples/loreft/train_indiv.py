import os
import torch
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq,
    set_seed,
    TrainingArguments
)
import wandb

import datetime
import json

import copy

from dataset import LoReftSupervisedDataset
from compute_metrics import compute_metrics

from pyreft import (
    TaskType,
    get_reft_model,
    LoreftIntervention,
    ReftDataCollator,
    ReftConfig,
    ReftTrainerForCausalLM
)


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
}
intervention_mapping = {
    "LoreftIntervention": LoreftIntervention,
}


def finetune(
    act_fn: str,
    add_bias: bool,
    model: str,
    layers: str,
    rank: int,
    position: str,
    epochs: int,
    seed: int,
    intervention_type: str,
    max_n_train_example: int,
    max_n_eval_example: int,
    is_wandb: bool,
    wandb_name: str,
    gradient_accumulation_steps: int,
    batch_size: int,
    output_dir: str,
    task: str,
    lr: float,
    schedule: str,
    data_dir: str,
    train_dataset: str,
    eval_dataset: str,
    save_model: bool,
    eval_batch_size: int,
    warmup_ratio: float,
    weight_decay: float,
    dropout: float,
    test_split: str,
    train_on_inputs: bool,
    max_length: int,
    use_normalized_template: bool,
    allow_cls_grad: bool,
    metric_for_best_model: str,
    dtype: str,
    logging_steps: int,
    wandb_dir: str,
    wandb_proj: str,
    share_weights: bool,
    greedy_decoding: bool,
    temperature: float,
    top_p: float,
    top_k: float,
    disable_reft: bool, # this will run baselines with LoRA only
    use_lora: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_modules: str,
    lora_layers: str,
    warm_init_vector_dir: str,
    args,
):
    assert task in {
        "commonsense", "math", "alpaca", "instruct", "ultrafeedback", "glue", "gsm8k",
        "ultrafeedback_pair", "commonsense_indiv"
    }

    dtype = dtype_mapping[dtype]
    
    # store/log run details
    print(
        f"task: {task}, model: {model}, intervention_type: {intervention_type}, "
        f"layers: {layers}, rank: {rank}, "
        f"position: {position}, epoch: {epochs}, train_on_inputs: {train_on_inputs}, "
        f"max_length: {max_length}, allow_cls_grad: {allow_cls_grad}"
    )
    set_seed(seed)

    model_name = model
    model_str = model.split("/")[-1]
    train_dataset_str = train_dataset
    
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if train_dataset is not None:
        run_name = f"{model_str}.{task}.{train_dataset_str}.{test_split}.{now}"
    else:
        run_name = f"{model_str}.{task}.{now}"

    # which layers to intervene on
    if layers.strip() == "":
        layers = []
    elif layers != "all":
        layers = [int(l) for l in layers.split(";")]
    else:
        temp_config = AutoConfig.from_pretrained(model)
        layers = [l for l in range(temp_config.num_hidden_layers)]

    unique_layers = copy.deepcopy(layers)
    if "+" in position and not share_weights:
        layers += layers

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False

    ReftDataset = LoReftSupervisedDataset
    train_dataset = ReftDataset(
        task, train_dataset_str, 
        tokenizer, data_split="train", seed=seed, max_n_example=max_n_train_example,
        **{"num_interventions": len(layers), "position": position, 
           "share_weights": share_weights, "test_split": test_split}
    )
    trigger_tokens = train_dataset.trigger_tokens
    num_labels = train_dataset.num_labels

    eval_dataset = ReftDataset(
        task, train_dataset_str,
        tokenizer, data_split=test_split, seed=seed, max_n_example=max_n_eval_example,
        **{"num_interventions": len(layers), "position": position, 
            "share_weights": share_weights}
    )
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=dtype if dtype != "float8" else None,  # save memory
        load_in_8bit=True if dtype == "float8" else False,
        device_map=device
    )
    config = model.config
    if need_resize:
        model.resize_token_embeddings(len(tokenizer))
    intervention_type = intervention_mapping[intervention_type]
    data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest"
        )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)
    intervention_dtype = torch.bfloat16 if isinstance(dtype, str) else dtype
    if warm_init_vector_dir != None:
        warm_init = torch.load(warm_init_vector_dir).to(device)
    else:
        warm_init = None
    representations = [{
        "layer": l, "component": f"base_model.model.model.layers[{l}].output" if use_lora else "block_output",
        "low_rank_dimension": rank,
        "intervention": intervention_type(
            embed_dim=config.hidden_size, low_rank_dimension=rank,
            dropout=dropout, dtype=intervention_dtype, act_fn=act_fn, device=device,
            add_bias=add_bias,
            warm_init_vector=warm_init[l,:] if warm_init != None else None,
        )
    } for l in layers]
    task_type=TaskType.CAUSAL_LM

    reft_config = ReftConfig(representations=representations)
    reft_model = get_reft_model(model, reft_config, set_device=not isinstance(dtype, str))
    reft_model.print_trainable_parameters()

    reft_model.model.train()
    n_params = reft_model.count_parameters(include_model=False)
    n_params_with_model = reft_model.count_parameters(include_model=True)

    if is_wandb:
        run = wandb.init(
            project=f"{wandb_proj}_{task}", 
            entity=wandb_name,
            name=run_name,
            dir=wandb_dir,
        )
        run.summary.update(vars(args))
        wandb.log(
            {"train/n_params": n_params, "train/n_params_with_model": n_params_with_model})

    # # training args
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch" if task == "glue" else "no",
        save_strategy="epoch" if task == "glue" else "no",
        metric_for_best_model=metric_for_best_model if task == "glue" else None,
        load_best_model_at_end=True if task == "glue" else False,
        logging_strategy="steps",
        save_total_limit=1, # for GLUE, it will save 2 at max.
        logging_steps=logging_steps,
        lr_scheduler_type=schedule,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        weight_decay=weight_decay,
        report_to="wandb" if is_wandb else "none",
        use_cpu=False if device == "cuda" else True,
        seed=seed,
        # until HF supports ReFT, this remains False! :)
        remove_unused_columns=False
    )

    # make trainer
    trainer_class = ReftTrainerForCausalLM
    trainer = trainer_class(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
    )
    if len(layers) > 0:
        trainer.train()

    # dump config
    args_dict = vars(args)
    args_dict["n_params"] = n_params
    json_file_name = f"{output_dir}/{run_name}/args.json"
    with open(json_file_name, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    # save model
    if save_model:
        reft_model.save(f"{output_dir}/{run_name}")

    # ensure everything is in eval mode
    reft_model.model.eval()
    for k,v in reft_model.interventions.items():
        _ = v[0].eval()

    print({"n_params": n_params})
    # do eval
    eval_results = {}


    generations, stats = compute_metrics(
        task, train_dataset_str, reft_model, tokenizer, eval_dataset, eval_dataset.raw_dataset,
        trigger_tokens, run_name, eval_batch_size, 
        None,
        test_split, greedy_decoding, temperature, top_p, top_k, is_base=len(layers)==0, base_model_trigger="Answer format: true/false"
    )

    # log
    eval_results.update(stats)
    if is_wandb:
        wandb.log(stats)
    generations = stats if generations is None else generations
    if "/" in train_dataset_str:
        train_dataset_str = train_dataset_str.replace("/", "_")
    result_json_file_name = f"{output_dir}/{run_name}/{train_dataset_str}_{test_split}_outputs.json"
    
    with open(result_json_file_name, 'w') as json_file:
        json.dump(generations, json_file, indent=4)

    result_json_file_name = f"{output_dir}/{run_name}/eval_results.json"
    eval_results["n_params"] = n_params
    with open(result_json_file_name, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)

    print(f"Training results can be found in {output_dir}/{run_name}")


def main():
    parser = argparse.ArgumentParser(description="A simple script that takes different arguments.")
    
    parser.add_argument('-task', '--task', type=str, default=None)
    parser.add_argument('-data_dir', '--data_dir', type=str, default="./datasets")
    parser.add_argument('-train_dataset', '--train_dataset', type=str, default=None)
    parser.add_argument('-eval_dataset', '--eval_dataset', type=str, default=None)
    parser.add_argument('-model', '--model', type=str, help='yahma/llama-7b-hf', default='yahma/llama-7b-hf')
    parser.add_argument('-seed', '--seed', type=int, help='42', default=42)
    parser.add_argument('-l', '--layers', type=str, help='2;10;18;26', default='2;10;18;26')
    parser.add_argument('-r', '--rank', type=int, help=8, default=8)
    parser.add_argument('-p', '--position', type=str, help='f1+l1', default='f1+l1')
    parser.add_argument('-e', '--epochs', type=int, help='1', default=1)
    parser.add_argument('-is_wandb', '--is_wandb', action='store_true')
    parser.add_argument('-wandb_name', '--wandb_name', type=str, default="reft")
    parser.add_argument('-save_model', '--save_model', action='store_true')
    parser.add_argument('-max_n_train_example', '--max_n_train_example', type=int, default=None)
    parser.add_argument('-max_n_eval_example', '--max_n_eval_example', type=int, default=None)
    parser.add_argument(
        '-type', '--intervention_type', type=str, 
        help='LoreftIntervention', default="LoreftIntervention")
    parser.add_argument('-gradient_accumulation_steps', '--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=4)
    parser.add_argument('-eval_batch_size', '--eval_batch_size', type=int, default=4)
    parser.add_argument('-output_dir', '--output_dir', type=str, default="./official_results")
    parser.add_argument('-lr', '--lr', type=float, default=5e-3)
    parser.add_argument('-schedule', '--schedule', type=str, default='linear')
    parser.add_argument('-wu', '--warmup_ratio', type=float, default=0.00)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.00)
    parser.add_argument('-act_fn', '--act_fn', type=str, default=None)
    parser.add_argument('-add_bias', '--add_bias', action='store_true')
    parser.add_argument('-test_split', '--test_split', type=str, default="validation")
    parser.add_argument('-train_on_inputs', '--train_on_inputs', action='store_true')
    parser.add_argument('-max_length', '--max_length', type=int, help=512, default=512)
    parser.add_argument('-nt', '--use_normalized_template', action='store_true')
    parser.add_argument('-allow_cls_grad', '--allow_cls_grad', action='store_true')
    parser.add_argument('-metric_for_best_model', '--metric_for_best_model', type=str, default="accuracy")
    parser.add_argument('-dtype', '--dtype', type=str, default="bfloat16" if device == "cuda" else "float32")
    parser.add_argument('-logging_steps', '--logging_steps', type=int, help=1, default=1)
    parser.add_argument('-wandb_dir', '--wandb_dir', type=str, default='wandb')
    parser.add_argument('-wandb_proj', '--wandb_proj', type=str, default='MyReFT')
    parser.add_argument('-sw', '--share_weights', action='store_true')
    parser.add_argument('-gd', '--greedy_decoding', action='store_true')

    # OUR ADDED PARAM
    parser.add_argument('-warm_init_vector_dir', '--warm_init_vector_dir', default=None, type=str)

    # decoding params
    parser.add_argument('-t', '--temperature', type=float, default=None)
    parser.add_argument('-top_p', '--top_p', type=float, default=None)
    parser.add_argument('-top_k', '--top_k', type=float, default=None)

    # lora add-ons
    parser.add_argument('-disable_reft', '--disable_reft', action='store_true')
    parser.add_argument('-use_lora', '--use_lora', action='store_true')
    parser.add_argument('-lora_rank', '--lora_rank', type=int, default=8)
    parser.add_argument('-lora_alpha', '--lora_alpha', type=int, default=32)
    parser.add_argument('-lora_modules', '--lora_modules', type=str, default="o_proj")
    parser.add_argument('-lora_layers', '--lora_layers', type=str, help='2;10;18;26', default='2;10;18;26')

    args = parser.parse_args()

    finetune(**vars(args), args=args)


if __name__ == "__main__":
    main()