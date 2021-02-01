# -*- coding: utf-8 -*-
# Copyright (c) 2020 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Katharina Ott, katharina.ott3@de.bosch.com

import torch.utils.data as data
from torchvision import datasets
from torchvision import transforms

from data.base_dataset import BaseDataset


class MNISTDataset(BaseDataset):
    image_size = 28

    def __init__(self):
        super(MNISTDataset, self).__init__()

    @staticmethod
    def get_properties():
        return {
            'image_size': MNISTDataset.image_size,
            'num_ip_ch': 1,
            'num_classes': 10,
        }

    def return_dataloader(self, split: str, phase: str, batch_size: int, serial_batches: bool, num_threads: int,
                          path_to_data='data/datasets/') -> data.DataLoader:
        # Data transforms
        all_transforms = transforms.Compose(
            [transforms.Resize(MNISTDataset.image_size), transforms.ToTensor()])

        dataset = datasets.MNIST(path_to_data, train=split == 'train',
                                 download=True, transform=all_transforms)

        dataloader = data.DataLoader(dataset, batch_size=batch_size,
                                     shuffle=not serial_batches,
                                     num_workers=num_threads,
                                     drop_last=phase == 'train'
                                     )
        return dataloader
