"""
The purpose of this file is to group and build all the Dataset objects, which will then be passed to the Datamodule.
"""

from torch.utils.data import Dataset
from typing import List, Optional

import torchvision.transforms.v2 as t
import torch
import os


class Augment(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        for_data: Optional[t.Transform],
        for_labels: Optional[t.Transform],
    ):
        self.dtst = dataset
        self.for_data = for_data
        self.for_labels = for_labels

    def __len__(self):
        return len(self.dtst)

    def __getitem__(self, index):
        image, label = self.dtst[index]
        if self.for_data:
            image = self.for_data(image)
        if self.for_labels:
            label = self.for_labels(label)
        return image, label


class DummyDtst(Dataset):
    def __init__(self, sample_size: List[int], dataset_size: int):
        super().__init__()
        self.sample_size = tuple(sample_size)
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return torch.rand(self.sample_size), torch.randint(0, 2, (1,)).float()
