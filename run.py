import yaml
import torch
from config import Param
from methods.utils import setup_seed
from methods.manager import Manager
import wandb
from dotenv import load_dotenv
import os
# os.environ['WANDB_MODE'] = 'disabled' # disable wandb for this script



def run(args):
    print(f"hyper-parameter configurations:")
    print(yaml.dump(args.__dict__, sort_keys=True, indent=4))

    setup_seed(args.seed)
    
    # for seed in [2421, 2021]:
        # args.seed = seed
    manager = Manager(args)
    manager.train(args)


if __name__ == "__main__":
    # Load configuration
    param = Param()
    args = param.args
    
    # wandb
    load_dotenv()
    
    wandb_api_key = "0806b2d5c00870a95f366d95c825d7680649abb7"  # Thay YOUR_WANDB_API_KEY bằng API key thực tế của bạn

    os.environ["WANDB_API_KEY"] = wandb_api_key
    
    wandb.login()

    # start a new wandb run to track this script
    if args.run_name is None:
        args.run_name = f"{args.dataname}_{args.seed}_{args.prompt_pool_size}_{args.prompt_length}_{args.prompt_top_k}"
        
    wandb.init(
        # set the wandb project where this run will be logged
        project="wave_eoe_tii",
        name = args.run_name,

        # track hyperparameters and run metadata
        config=args.__dict__,
    )

    # Device
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)

    # Num GPU
    args.n_gpu = torch.cuda.device_count()

    # Task name
    args.task_name = args.dataname

    # rel_per_task
    if args.dataname == "FewRel":
        args.rel_per_task = 8
    elif args.dataname == "TACRED":
        args.rel_per_task = 4
    else:
        raise ValueError("Invalid dataname")

    # Run
    run(args)
    
    wandb.finish()
