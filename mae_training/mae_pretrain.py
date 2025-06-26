# standard imports
import json
import munch
from munch import Munch
import numpy as np
import os
import random
import sys
import toml

# PyTorch/ML imports
import torch
import timm.optim.optim_factory as optim_factory
import wandb

# local imports
sys.path.insert(0, '/share/home/conradb/git/hephaestus-minicubes-ssl/')
import utilities.utils as utils
import mae_training.mae.models_mae as models_mae
from mae_training.mae.utils import NativeScalerWithGradNormCount as NativeScaler, load_model
from mae_training.mae.engine_pretrain import train_one_epoch

def load_global_config(filepath:str="project_config.toml"):
    return munch.munchify(toml.load(filepath))

def munch_to_dict(obj):
    if isinstance(obj, Munch):
        return {k: munch_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [munch_to_dict(i) for i in obj]
    else:
        return obj

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set:
    torch.backends.cudnn.deterministic = True # Use deterministic algorithms.
    torch.backends.cudnn.benchmark = False # Causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def create_model(config: dict) -> torch.nn.Module:
    model = models_mae.__dict__[config.model.model](img_size=config.dataloader.image_size, norm_pix_loss=config.model.norm_pix_loss)
    return model


def pretrain(configs, verbose):

    configs = load_global_config(configs)

    # print(json.dumps(munch_to_dict(configs), indent=2))

    if verbose:
        print('='*20)
        print('Initializing MAE pretraining')
        print('='*20)

    if configs.wandb.wandb:
        wandb_id = wandb.util.generate_id()

        wandb.init(
            project=configs.wandb.wandb_project,
            entity=configs.wandb.wandb_entity,
            config=munch_to_dict(configs),
            resume=False,
            id=wandb_id,
            reinit=True
        )

    print('here')

    if configs.wandb.wandb:
        configs.model.checkpoint_path = str(utils.create_checkpoint_directory(configs, wandb_run=wandb.run))
    else:
        configs.model.checkpoint_path = str(utils.create_checkpoint_directory(configs))

    if not configs.wandb.wandb:
        # Save model configs in checkpoint directory
        configs_path = os.path.join(configs.model.checkpoint_path,'configs.json')
        if not os.path.exists(configs_path):
            with open(str(configs_path), 'wb') as f:
                json.dump(configs, f, indent=3)

    train_loader, _, _ = utils.create_webdataset_dataloaders(configs)

    print(f'Cuda available = {torch.cuda.is_available()}')
    set_seed(configs.dataloader.seed)
    device = torch.device(configs.train.device)

    # Load model
    model = create_model(configs)
    model.to(device)

    # Scale learning rate, we are using 1 GPU (not distributed training) therefore not really necessary
    configs.train.lr = configs.train.blr * configs.dataloader.batch_size / 256
    print("base lr: %.2e" % (configs.train.lr * 256 / configs.dataloader.batch_size))
    print("actual lr: %.2e" % configs.train.lr)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model, configs.train.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=configs.train.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = NativeScaler()
    # load_model(configs, model, optimizer, loss_scaler)

    print('='*20)
    print(f"Starting MAE petraining for {configs.train.epochs} epochs")
    print('='*20)

    for epoch in range(0, configs.train.epochs):
        train_stats, lr, image_logs = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=None,
            config=configs
        )

        print(f'Epoch {epoch}/{configs.train.epochs}, Batch Loss is {train_stats}, end of epoch LR is {lr}')

        log_dict = {'Epoch': epoch, 'Pretrain loss': train_stats, 'LR': lr}
        
        if configs.wandb.wandb:
            wandb.log(log_dict)
            if (epoch+1) % 20 == 0:
                wandb.log(image_logs)
                
        if (epoch+1) % 10 == 0:
            save_dict = {
                'epoch': epoch,
                'loss' : train_stats,
                'arch' : configs.model.model,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            torch.save(save_dict,os.path.join(configs.model.checkpoint_path,str(configs.model.model)+'_epochs_'+str(configs.train.epochs)+'.pt'))
            if verbose:
                print('Saving checkpoint')
            if configs.wandb.wandb:
                if (epoch+1) == configs.train.epochs: 
                    artifact = wandb.Artifact(configs.model.model+'_epochs_'+str(configs.train.epochs)+'_blr_'+str(configs.train.blr), type='model')
                    artifact.add_file(os.path.join(configs.model.checkpoint_path,str(configs.model.model)+'_epochs_'+str(configs.train.epochs)+'.pt'))
                    wandb.log_artifact(artifact)

    wandb.finish(exit_code=0)

if __name__ == '__main__':
    pretrain('../configs/pretrain_config.toml', True)
