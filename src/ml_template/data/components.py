"""
The purpose of this file is to group and build all the Dataset objects, which will then be passed to the Datamodule.
"""
from typing import List, Optional, Sized
from torch.utils.data import Dataset
from abc import ABC

import torchvision.transforms.v2 as t
import torchvision.tv_tensors as tv
import torch

class SizedDataset(Dataset, Sized, ABC):
    ...

class AugmentImageData(SizedDataset):
    def __init__(
        self,
        dataset: SizedDataset,
        transforms: Optional[t.Transform],
    ):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image, label = self.transforms(image, label)
        return image, label

class DummyDataset(SizedDataset):
    def __init__(self, sample_size: List[int], dataset_size: int):
        super().__init__()
        self.sample_size = tuple(sample_size)
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return tv.Image(torch.rand(self.sample_size)), torch.randint(0, 2, (1,)).float()
