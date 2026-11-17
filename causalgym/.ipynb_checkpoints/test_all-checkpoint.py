#from data import list_datasets
from das import experiment
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import WEIGHTS
import torch
import os

def run_command(
    tokenizer: AutoTokenizer,
    gpt: AutoModelForCausalLM,
    model_name: str,
    dataset: str,
    lr: float,
    only_das: bool,
    hparam_non_das: bool,
    das_label: str,
    revision: str,
    folder: str,
    manipulate: str,
):
    experiment(
        model=model_name,
        dataset=dataset,
        steps=100,
        eval_steps=100,
        grad_steps=1,
        batch_size=4,
        intervention_site="block_output",
        strategy="last",
        lr=lr,
        only_das=True,
        hparam_non_das=hparam_non_das,
        das_label=das_label,
        revision=revision,
        log_folder=folder,
        manipulate=manipulate,
        tokenizer=tokenizer,
        gpt=gpt,
    )

def main(
    model: str, lr: float=5e-3, hparam_non_das: bool=False, only_das: bool=True,
    das_label: str=None, start: int=None, end: int=None, folder: str="das", revision: str="main",
    manipulate: str=False, dataset: str=None, template: str=""):


    path = f"/srv/data1/abhidipbhatt/output/DAS/{model}"
    model_path = f"/srv/data1/abhidipbhatt/models/GPT2/{model}-FT_32B"
    log_folder = os.path.join(path, folder)
    
    # load model + tokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        #revision=revision,
        #torch_dtype= torch.float32,
    ).to(device)

    
   
    # make folder
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    #folder = f"/srv/data1/abhidipbhatt/output/DAS/{folder}"
    #for dataset in datasets[start:end]:
    run_command(tokenizer, gpt, model_path, dataset, lr, only_das, hparam_non_das, das_label, revision, log_folder, manipulate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-medium")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--only-das", action="store_true")
    parser.add_argument("--hparam_non_das", action="store_true")
    parser.add_argument("--das-label", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--folder", type=str, default="log")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--manipulate", type=str, default=None)
    #parser.add_argument("--dataset", type=str, default='not_not', help='all|AB|A_not|B_not|not_not')
    #parser.add_argument("--datasets", nargs='+', default=None)
    # == Modified to accept template as a list of strings ==
    #parser.add_argument("--template", nargs='+', default=None)
    # == End modified ==
    args = parser.parse_args()
    for d in ['all','AB','A_not','B_not','not_not']:
        args.dataset = d
        main(**vars(args))