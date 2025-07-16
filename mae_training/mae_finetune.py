# standard imports
import json
import munch
from munch import Munch
import numpy as np
import os
import random
import sys
import toml
import time

# PyTorch/ML imports
import torch
from timm.layers import trunc_normal_
import wandb

# local imports
sys.path.insert(0, '/share/home/conradb/git/hephaestus-minicubes-ssl/')
import utilities.utils as utils
import mae_training.mae.models_vit as models_vit
from mae_training.mae.pos_embed import interpolate_pos_embed
from mae_training.mae.utils import LARS, NativeScalerWithGradNormCount as NativeScaler, RandomResizedCrop, param_groups_lrd
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
    if config.model.stage == 'linprobe':
        model = models_vit.__dict__[config.model.model](img_size=config.dataloader.image_size, num_classes=config.dataloader.num_classes, global_pool=config.model.global_pool)
    elif config.model.stage == 'e2e-finetuning':
        model = models_vit.__dict__[config.model.model](img_size=config.dataloader.image_size, num_classes=config.dataloader.num_classes, drop_path_rate=config.model.drop_path_rate, global_pool=config.model.global_pool)
    else:
        print('Finetuning stage is not set')
        
    return model

def finetune(configs, verbose):

    configs = load_global_config(configs)

    # print(json.dumps(munch_to_dict(configs), indent=2))

    if verbose:
        print('='*20)
        if configs.model.stage == 'linprobe':
            print('Initializing MAE linear probing')
        else:
            print('Initializing MAE end to end finetuning')
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

    if configs.model.stage == 'linprobe':

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

        # for linear prob only
        # # hack: revise model's head with BN
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True
    else:
        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters))

    configs.train.lr = configs.train.blr * configs.dataloader.batch_size / 256
    print("base lr: %.2e" % (configs.train.lr * 256 / configs.dataloader.batch_size))
    print("actual lr: %.2e" % configs.train.lr)
    
    if configs.model.stage == 'linprobe':
        optimizer = LARS(model.head.parameters(), lr=configs.train.lr, weight_decay=configs.train.weight_decay)
        
    else:
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = param_groups_lrd(
            model, configs.train.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),
            layer_decay=configs.model.layer_decay
            )
            
        optimizer = torch.optim.AdamW(param_groups, lr=configs.train.lr)

    
    print(optimizer)
    loss_scaler = NativeScaler()

    # either CE or Focal Loss
    criterion = utils.init_criterion(configs)

    if configs.model.stage == 'linprobe':
        print(f"Starting MAE linear probing training for {configs.train.epochs} epochs")
    else:
        print(f"Starting MAE end to end finetuning training for {configs.train.epochs} epochs")

    start = time.time()
    for epoch in range(0, configs.train.epochs):
        
        train_log_dict, train_image_logs = train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=None,
            config=configs)

        print(train_log_dict)

        val_log_dict = evaluate(val_loader, model, criterion, configs.train.device, phase='Val', epoch=epoch)

        if (epoch+1) % 10 == 0:
            test_dict = evaluate(test_loader, model, criterion, configs.train.device, phase='Test')

            print(f'Test Acc (Non-Def) is: {test_dict["MulticlassAccuracy"][0]}')
            print(f'Test Acc (Def) is: {test_dict["MulticlassAccuracy"][1]}')
            print(f'Test F1 Score (Non-Def) is: {test_dict["MulticlassF1Score"][0]}')
            print(f'Test F1 Score (Def) is: {test_dict["MulticlassF1Score"][1]}')

            save_dict = {
                'epoch': epoch,
                'loss' : train_log_dict['train_loss'],
                'arch' : configs.model.model,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            torch.save(save_dict,os.path.join(configs.model.checkpoint_path,str(configs.model.model)+'_epochs_'+str(configs.train.epochs)+'_blr_'+str(configs.train.blr)+'_'+configs.train.criterion+'_'+configs.model.stage+'.pt'))
            if verbose:
                print('Saving checkpoint')

        if configs.wandb.wandb:
            wandb.log({
                configs.train.criterion+"/train": train_log_dict['train_loss'],
                configs.train.criterion+"/val": val_log_dict['val_loss'],
                "non_def/f1_score/train": train_log_dict['Train MulticlassF1Score Class: Non Deformation'].item(),
                "non_def/f1_score/val": val_log_dict['Val MulticlassF1Score Class: Non Deformation'].item(),
                "def/f1_score/train": train_log_dict['Train MulticlassF1Score Class: Deformation'].item(),
                "def/f1_score/val": val_log_dict['Val MulticlassF1Score Class: Deformation'].item(),
                "non_def/accuracy/val": val_log_dict['Val MulticlassAccuracy Class: Non Deformation'].item(),
                "def/accuracy/val": val_log_dict['Val MulticlassAccuracy Class: Deformation'].item(),
                "non_def/precision/val": val_log_dict['Val MulticlassPrecision Class: Non Deformation'].item(),
                "def/precision/val": val_log_dict['Val MulticlassPrecision Class: Deformation'].item(),
                "non_def/recall/val": val_log_dict['Val MulticlassRecall Class: Non Deformation'].item(),
                "def/recall/val": val_log_dict['Val MulticlassRecall Class: Deformation'].item(),
            }, step=epoch)
            if (epoch+1) % 10 == 0:
                wandb.log(train_image_logs, step=epoch)

        if configs.wandb.wandb:
            if (epoch+1) == configs.train.epochs:
                table = wandb.Table(columns=["metric", "Non Deformation", "Deformation"])
                for metric, values in test_dict.items():
                    table.add_data(metric, *values.detach().cpu().numpy())
                wandb.log({"Final Test Set Metrics": table})

                artifact = wandb.Artifact(configs.model.model+'_epochs_'+str(configs.train.epochs)+'_blr_'+str(configs.train.blr)+'_'+configs.train.criterion+'_'+configs.model.stage, type='model')
                artifact.add_file(os.path.join(configs.model.checkpoint_path,str(configs.model.model)+'_epochs_'+str(configs.train.epochs)+'_blr_'+str(configs.train.blr)+'_'+configs.train.criterion+'_'+configs.model.stage+'.pt'))
                wandb.log_artifact(artifact)
        
    wandb.finish(exit_code=0)
    end = time.time()
    elapsed = end - start
    print('='*20)
    if configs.model.stage == 'linprobe':
        print(f"MAE linear probing complete in {elapsed} seconds for {configs.train.epochs} epochs")
    else:
        print(f"MAE end to end finetuning complete in {elapsed} seconds for {configs.train.epochs} epochs")
    print('='*20)

if __name__ == '__main__':
    finetune('../configs/linprobe_config.toml', True)