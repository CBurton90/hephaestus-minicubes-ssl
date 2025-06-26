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
from timm.layers import trunc_normal_
import wandb

# local imports
sys.path.insert(0, '/share/home/conradb/git/hephaestus-minicubes-ssl/')
import utilities.utils as utils
import mae_training.mae.models_vit as models_vit
from mae_training.mae.pos_embed import interpolate_pos_embed
from mae_training.mae.utils import LARS, NativeScalerWithGradNormCount as NativeScaler, RandomResizedCrop
from mae_training.mae.engine_finetune import train_one_epoch, evaluate

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
    model = models_vit.__dict__[config.model.model](img_size=config.dataloader.image_size, num_classes=config.dataloader.num_classes, global_pool=config.model.global_pool)
    return model

def linprobe(configs, verbose):

    configs = load_global_config(configs)

    # print(json.dumps(munch_to_dict(configs), indent=2))

    if verbose:
        print('='*20)
        print('Initializing MAE linear probing')
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

    train_loader, val_loader, test_loader = utils.create_webdataset_dataloaders(configs)

    print(f'Cuda available = {torch.cuda.is_available()}')
    set_seed(configs.dataloader.seed)
    device = torch.device(configs.train.device)

    # Load model
    model = create_model(configs)

    checkpoint = torch.load(configs.model.checkpoint_load_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % configs.model.checkpoint_load_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    if configs.model.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    # manually initialize fc layer: following MoCo v3
    trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters))

    configs.train.lr = configs.train.blr * configs.dataloader.batch_size / 256
    print("base lr: %.2e" % (configs.train.lr * 256 / configs.dataloader.batch_size))
    print("actual lr: %.2e" % configs.train.lr)

    optimizer = LARS(model.head.parameters(), lr=configs.train.lr, weight_decay=configs.train.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print(f"Starting linprobe training for {configs.train.epochs} epochs")

    for epoch in range(0, configs.train.epochs):
        
        train_loss = train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=None,
            config=configs)

        print(train_loss)

if __name__ == '__main__':
    linprobe('../configs/linprobe_config.toml', True)