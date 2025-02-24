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
    
    wandb.login()

    # start a new wandb run to track this script
    if args.run_name is None:
        # --run_name $dataname-$seed-$encoder_epochs-$prompt_pool_epochs-$classifier_epochs-$prompt_pool_size-$prompt_length-$prompt_top_k-$num_descriptions-$beta
        args.run_name = f"{args.dataname}-{args.seed}-{args.encoder_epochs}-{args.prompt_pool_epochs}-{args.classifier_epochs}-{args.prompt_pool_size}-{args.prompt_length}-{args.prompt_top_k}-{args.num_descriptions}-{args.beta}"
        
    wandb.init(
        # set the wandb project where this run will be logged
        project="wave",
        name = f"{args.dataname}-{args.seed}",

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
