# hephaestus-minicubes-ssl
Exploring self-supervised computer vision models for the Hephaestsus Minicubes benchmark (Papadopoulos et al., 2025)

https://arxiv.org/abs/2505.17782

https://github.com/Orion-AI-Lab/Hephaestus-minicubes

## Downloading and webdataset dataloaders

Use `dl_all.sh` to download the Zarr files from Dropbox. The individual file download may cut out around the 50Gb mark so running again may be necessary (but the curl `-C -` flag should pick it up where it left off).

Set webdataset dataloader config under the appropriate config file (e.g. `configs/pretrain_config.toml`).

Run `mae_training/mae_pretrain.py` three times to create the train, val, test sharded tar directories for the respective webdatasets.

Run `webdataset_renaming.sh` to rename tar files for the train, val, test webdataset dataloaders.

## Masked Autoencoder (He et al., 2021)

https://arxiv.org/abs/2111.06377

### Pretraining

Pretraining date range is set to encompass the train and val sets withing the Hephaestus minicubes paper, the test date range is left unseen. A masking ratio of 60% is used and a ViT Large model is trained for 800 epochs with 40 warmup epochs, the timeseries length is kept at 1 and only interferometric phase, coherence, and DEM channels are used for pretraining. Further hyperparameters can be found in the config file `configs/pretrain_config.toml`.

Run with `mae_training/mae_pretrain.py`.

| Modality       | Masked       | Reconstruction    | Original     |
|-------------|-------------------|--------------|-----------------|
| Phase | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/5fc519f2-b614-4afd-805a-ed5ffeed7276" /> | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/de305f6c-0d3c-4e37-8f34-5528ba6afde2" /> | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/ed79b506-d74f-4f9e-950f-d263c1b2efe1" /> |
| Coherence | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/2f9c0756-1fe8-42db-a216-5402d6c11187" /> | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/fab2776a-b023-4942-a36b-b2ae109bcb6f" /> | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/9d36cd84-18ce-46bc-ba89-51c10e18fdaf" /> |
| DEM | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/f74950e7-fe82-4513-8d6e-1c64480e13bf" /> | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/0252683b-77c6-42cd-9eda-26962873ab2e" /> | <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/d81c2cac-b084-4fb1-981e-5467be958ea9" /> |

### Post-training

Train, val, test sets matching Hephaestus minicubes paper. All metrics reported are for the deformation class (1).

Augmentations: 
- RRC = Random Resized Crop
- VF = Vertical Flip
- HF = Horizontal Flip
- RR = Random Rotation

#### End to end finetuning results

End to end finetuning of ViT Large trained for 50 epochs with 5 warmup epochs and batch size of 28 (learning rate scaling is used as per MAE paper).

| Loss function | Base Learning Rate| Weight Decay| Augs| Prec | Rec | F1 | AUROC |
|----------|---------|-------|---------|--------|------|--------|-------|
| Cross Entropy | 1e-3 | 0.05| RRC/VF/HF/RR | **0.7976** | **0.7976** | **0.7976** | 0.9528 |
| Focal Loss    | 1e-3 | 0.07 | RRC/VF/HF/RR | 0.7688 | 0.7839 | 0.7763 | **0.9567** |

#### Linear probing results

Linear probing of ViT Large with frozen weights trained for 90 epochs with 10 warmup epochs and batch size of 112 (learning rate scaling is used as per MAE paper).

| Loss function | Base Learning Rate| Weight Decay| Augs| Prec | Rec | F1 | AUROC |
|----------|---------|-------|---------|--------|------|--------|-------|
| Cross Entropy | 2 | 0 | RRC/VF/HF/RR | **0.5722** | 0.6149 | 0.5928 | **0.8602** |
| Focal Loss    | 2 | 0 | RRC/VF/HF/RR | 0.5336 | **0.7014** | **0.6061** | 0.8472 |


