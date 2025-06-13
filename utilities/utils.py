import glob
import io
import json
import munch
import numpy as np
import os
import random
import toml

# PyTorch/webdataset imports
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import webdataset as wds

# Local imports
import webdataset_utils as wds_utils

def load_global_config(filepath:str="project_config.toml"):
    return munch.munchify(toml.load(filepath))

def get_dataset(config, mode="train", webdataset_write= False, verbose=False):
    return InSarDataset(config, mode, verbose=verbose, webdataset_write=webdataset_write)

def normalize(image_timeseries, config, statistics_path="/home/conradb/git/hephaestus-minicubes-ssl/statistics.json"):
    """
    Normalize only the channels present in the image, based on the config.
    Handles geomorphology and atmospheric channels in primary/secondary pairs.

    Args:
        image_timeseries (Tensor): shape [T, C, H, W]
        config (dict): includes 'geomorphology_channels' and 'atmospheric_channels'
        statistics_path (str): JSON file with per-channel stats

    Returns:
        Tensor: normalized image_timeseries
    """
    statistics = json.load(open(statistics_path, "r"))

    means = []
    stds = []

    # 1. Geomorphology channels come first
    for ch in config.dataloader.geomorphology_channels:
        if ch in statistics:
            means.append(statistics[ch]["mean"])
            stds.append(statistics[ch]["std"])
        else:
            print(f"Missing stats for geomorph channel: {ch}")

    # 2. Atmospheric channels in primary-secondary pairs
    for ch in config.dataloader.atmospheric_channels:
        if ch in statistics:
            # Add twice: once for primary, once for secondary
            means.extend([statistics[ch]["mean"], statistics[ch]["mean"]])
            stds.extend([statistics[ch]["std"], statistics[ch]["std"]])
        else:
            print(f"Missing stats for atmospheric variable: {ch}")

    # Apply normalization over each channel
    for i in range(len(means)):
        image_timeseries[:, i, :, :] = (image_timeseries[:, i, :, :] - means[i]) / stds[i]

    return image_timeseries

# run 3 times for train, val, test dataloaders
def create_webdataset_dataloaders(configs, repeat=False, resample_shards=False):

    configs = load_global_config(configs)

    print(configs.dataloader.train_years[0])
    
    random.seed(configs.dataloader.seed)
    np.random.seed(configs.dataloader.seed)

    all_channels = [
        "insar_difference",
        "insar_coherence",
        "dem",
        "primary_date_total_column_water_vapour",
        "secondary_date_total_column_water_vapour",
        "primary_date_surface_pressure",
        "secondary_date_surface_pressure",
        "primary_date_vertical_integral_of_temperature",
        "secondary_date_vertical_integral_of_temperature"
        ]

    def get_channel_indices(channel_list, all_channels, is_atmospheric=False):
        indices = []
        for channel in channel_list:
            if is_atmospheric:
                prim = f"primary_date_{channel}"
                sec = f"secondary_date_{channel}"
                if prim in all_channels:
                    indices.append(all_channels.index(prim))
                else:
                    print(f"Warning: {prim} not in all_channels")
                if sec in all_channels:
                    indices.append(all_channels.index(sec))
                else:
                    print(f"Warning: {sec} not in all_channels")
            else:
                if channel in all_channels:
                    indices.append(all_channels.index(channel))
                else:
                    print(f"Warning: {channel} not in all_channels")
        return indices

    def get_patches(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"])).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]))
            sample = torch.load(io.BytesIO(sample["sample.pth"]))
            if isinstance(label, dict):
                label = label["label"]

            image = image.reshape(configs.dataloader.timeseries_length, len(all_channels), configs.dataloader.image_size, configs.dataloader.image_size)
            label = label.reshape(configs.dataloader.timeseries_length, 1, configs.dataloader.image_size, configs.dataloader.image_size)

            # Select only the relevant geomorphology channels and atmospheric channels
            selected_channels = geomorphology_indices + atmospheric_indices
            image = image[:, selected_channels, :, :]  # Keep only the selected channels

            image = normalize(image, configs)
            
            if configs.dataloader.task == 'segmentation':
                if configs.dataloader.timeseries_length != 1:
                    if configs.dataloader.mask_target == 'peak':
                        counts = torch.sum(label, dim=(2, 3))
                        label = label[torch.argmax(counts)]
                    elif configs.dataloader.mask_target == 'last':
                        label = label[-1, :, :, :]
                    elif configs.dataloader.mask_target == 'union':
                        label = torch.sum(label, dim=0)
                        label = torch.where(label > 0, 1, 0)
                else:
                    label = label[-1, :, :, :] # Last channel
            else:
                label = torch.tensor(int(np.any(sample['label'])==1))

            image = image.reshape(configs.dataloader.timeseries_length*(len(configs.dataloader.geomorphology_channels)+2*len(configs.dataloader.atmospheric_channels)), configs.dataloader.image_size, configs.dataloader.image_size)

            if configs.dataloader.task == 'segmentation':
                if configs.dataloader.mask_target == 'all':
                    label = label.reshape(configs.dataloader.timeseries_length, configs.dataloader.image_size, configs.dataloader.image_size)
                else:
                    label = label.reshape(configs.dataloader.image_size, configs.dataloader.image_size)

            yield (image, label, sample)

    def get_patches_eval(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"])).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]))
            sample = torch.load(io.BytesIO(sample["sample.pth"]))
            if isinstance(label, dict):
                label = label["label"]

            image = image.reshape(configs.dataloader.timeseries_length, len(all_channels), configs.dataloader.image_size, configs.dataloader.image_size)
            label = label.reshape(configs.dataloader.timeseries_length, 1, configs.dataloader.image_size, configs.dataloader.image_size)

            # Select only the relevant geomorphology channels and atmospheric channels
            selected_channels = geomorphology_indices + atmospheric_indices
            image = image[:, selected_channels, :, :]  # Keep only the selected channels

            image = normalize(image, configs)
            
            if configs.dataloader.task == 'segmentation':
                if configs.dataloader.timeseries_length != 1:
                    if configs.dataloader.mask_target == 'peak':
                        counts = torch.sum(label, dim=(2, 3))
                        label = label[torch.argmax(counts)]
                    elif configs.dataloader.mask_target == 'last':
                        label = label[-1, :, :, :]
                    elif configs.dataloader.mask_target == 'union':
                        label = torch.sum(label, dim=0)
                        label = torch.where(label > 0, 1, 0)
                else:
                    label = label[-1, :, :, :] # Last channel
            else:
                label = torch.tensor(int(np.any(sample['label'])==1))

            image = image.reshape(configs.dataloader.timeseries_length*(len(configs.dataloader.geomorphology_channels)+2*len(configs.dataloader.atmospheric_channels)), configs.dataloader.image_size, configs.dataloader.image_size)

            if configs.dataloader.task == 'segmentation':
                if configs.dataloader.mask_target == 'all':
                    label = label.reshape(configs.dataloader.timeseries_length, configs.dataloader.image_size, configs.dataloader.image_size)
                else:
                    label = label.reshape(configs.dataloader.image_size, configs.dataloader.image_size)

            yield (image, label, sample)

    geomorphology_indices = get_channel_indices(configs.dataloader.geomorphology_channels, all_channels)
    atmospheric_indices = get_channel_indices(configs.dataloader.atmospheric_channels, all_channels, is_atmospheric=True)

    print(geomorphology_indices) # to be removed
    print(atmospheric_indices) # to be removed
    
    configs.dataloader.webdataset_path = os.path.join(configs.dataloader.webdataset_root, str(configs.dataloader.timeseries_length))
    print(configs.dataloader.webdataset_path) # to be removed
    
    for mode in ["train", "val", "test"]:
        if mode == "train":
            if not os.path.isdir(os.path.join(configs.dataloader.webdataset_path, 'train_pos')) or not os.path.isdir(os.path.join(configs.dataloader.webdataset_path, 'train_neg')):
                wds_utils.wds_write_parallel(configs, mode)
                print(f"Created webdataset for: {mode}")
                exit(1)
        else:
            if not os.path.isdir(os.path.join(configs.dataloader.webdataset_path, mode)):
                wds_utils.wds_write_parallel(configs, mode)
                print("Created webdataset for: ", mode)
                exit(1)

    compress = configs.dataloader.get("compress", False)
    ext = ".tar.gz" if compress else ".tar"

    # create train dataloader

    max_train_pos_shard = np.sort(glob.glob(os.path.join(configs.dataloader.webdataset_path, "train_pos", f"*{ext}")))[-1]
    print(max_train_pos_shard)
    max_train_pos_index = max_train_pos_shard.split("-train_pos-")[-1][:-4]
    print(max_train_pos_index)
    max_train_neg_shard = np.sort(glob.glob(os.path.join(configs.dataloader.webdataset_path, "train_neg", f"*{ext}")))[-1]
    max_train_neg_index = max_train_neg_shard.split("-train_neg-")[-1][:-4]

    pos_train_shards = os.path.join(configs.dataloader.webdataset_path, "train_pos", "sample-train_pos-{000000.." + max_train_pos_index + "}"+ext,)

    print(pos_train_shards)

    neg_train_shards = os.path.join(configs.dataloader.webdataset_path, "train_neg", "sample-train_neg-{000000.." + max_train_neg_index + "}"+ext,)

    positives = wds.WebDataset(pos_train_shards, shardshuffle=1, resampled=False).shuffle(configs.dataloader.webdataset_shuffle_size).compose(get_patches)
    negatives = wds.WebDataset(neg_train_shards, shardshuffle=1, resampled=False).shuffle(configs.dataloader.webdataset_shuffle_size).compose(get_patches)

    count_pos = len([iter(positives)])
    count_neg = len([iter(negatives)])
    train_dataset = RandomMix(datasets=[positives, negatives], probs=[1/count_pos, 1/count_neg])

    train_loader = wds.WebLoader(
        train_dataset,
        num_workers=configs.dataloader.num_workers,
        batch_size=None,
        shuffle=False,
        pin_memory=False,
        prefetch_factor=configs.dataloader.prefetch_factor,
        persistent_workers=configs.dataloader.persistent_workers,
    ).shuffle(configs.dataloader.webdataset_shuffle_size).batched(configs.dataloader.batch_size, partial=False)

    train_loader = (
        train_loader.unbatched()
        .shuffle(
            configs.dataloader.webdataset_shuffle_size,
            initial=configs.dataloader.webdataset_initial_buffer,
        )
        .batched(configs.dataloader.batch_size)
    )

    if repeat:
        train_loader = train_loader.repeat()

    # create val dataloader

    max_val_shard = np.sort(glob.glob(os.path.join(configs.dataloader.webdataset_path, "val", f"*{ext}")))[-1]
    max_val_index = max_val_shard.split("-val-")[-1][:-4]

    val_shards = os.path.join(
        configs.dataloader.webdataset_path,
        "val",
        "sample-val-{000000.." + max_val_index + "}" + ext,
    )

    val_dataset = wds.WebDataset(val_shards, shardshuffle=False, resampled=False)
    val_dataset = val_dataset.compose(get_patches_eval)
    val_dataset = val_dataset.batched(configs.dataloader.batch_size, partial=True)

    val_loader = wds.WebLoader(
        val_dataset,
        num_workers=configs.dataloader.num_workers,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
    )

    # create test dataloader

    max_test_shard = np.sort(glob.glob(os.path.join(configs.dataloader.webdataset_path, "test", f"*{ext}")))[-1]
    max_test_index = max_test_shard.split("-test-")[-1][:-4]
    test_shards = os.path.join(
        configs.dataloader.webdataset_path,
        "test",
        "sample-test-{000000.." + max_test_index + "}"+ext,
    )

    test_dataset = wds.WebDataset(test_shards, shardshuffle=False, resampled=False)
    test_dataset = test_dataset.compose(get_patches_eval)
    test_dataset = test_dataset.batched(configs.dataloader.batch_size, partial=True)

    test_loader = wds.WebLoader(
        test_dataset,
        num_workers=configs.dataloader.num_workers,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class RandomMix(IterableDataset):
    """Iterate over multiple datasets by randomly selecting samples based on given probabilities."""

    def __init__(self, datasets, probs=None, longest=False):
        """Initialize the RandomMix iterator.

        Args:
            datasets (list): List of datasets to iterate over.
            probs (list, optional): List of probabilities for each dataset. Defaults to None.
            longest (bool): If True, continue until all datasets are exhausted. Defaults to False.
        """
        self.datasets = datasets
        self.probs = probs
        self.longest = longest

    def __iter__(self):
        """Return an iterator over the sources.

        Returns:
            iterator: An iterator that yields samples randomly from the datasets.
        """
        sources = [iter(d) for d in self.datasets]
        return random_samples(sources, self.probs, longest=self.longest)


def random_samples(sources, probs=None, longest=False):
    """Yield samples randomly from multiple sources based on given probabilities.

    Args:
        sources (list): List of iterable sources to draw samples from.
        probs (list, optional): List of probabilities for each source. Defaults to None.
        longest (bool): If True, continue until all sources are exhausted. Defaults to False.

    Yields:
        Sample randomly selected from one of the sources.
    """
    if probs is None:
        probs = [1] * len(sources)
    else:
        probs = list(probs)
    while len(sources) > 0:
        cum = (np.array(probs) / np.sum(probs)).cumsum()
        r = random.random()
        i = np.searchsorted(cum, r)

        try:
            yield next(sources[i])
        except StopIteration:
            if longest:
                del sources[i]
                del probs[i]
            else:
                break






if __name__ == '__main__':
    train, val, test = create_webdataset_dataloaders('../configs/config.toml')

    for image, label, sample in val:
        print(label)


    