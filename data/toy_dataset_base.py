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

import os
import re
import typing

import torch
import torch.utils.data as data

from data.base_dataset import BaseDataset
from data.sphere_dataset_generation.generate_sphere_dataset import (
    generate_sphere_dataset,
)
from data.three_minima_dataset_generation.data_generation import generate_three_minima_dataset


def load_obj(name, path):
    dict_file = os.path.join(path, name)
    data = torch.load(dict_file)
    return data


class Dataset(data.Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        x = self.data[item, :]
        y = self.labels[item]
        return x, y


class ToyDataset(BaseDataset):
    def __init__(self):
        super(ToyDataset, self).__init__()
        self.dataset_name = ""

    @staticmethod
    def get_properties() -> typing.Dict[str, int]:
        raise NotImplementedError

    def load_data_train(self, path: str) -> data.Dataset:
        file_name = "train_data.pt"
        path = os.path.join(path, self.dataset_name)
        data_dict = load_obj(file_name, path)
        data = data_dict["data"]
        labels = data_dict["labels"]
        train_data = Dataset(data, labels)
        return train_data

    def load_data_test(self, path: str) -> data.Dataset:
        file_name = "test_data.pt"
        path = os.path.join(path, self.dataset_name)
        data_dict = load_obj(file_name, path)
        data = data_dict["data"]
        labels = data_dict["labels"]
        test_data = Dataset(data, labels)
        return test_data

    def create_if_not_exist(self, path):
        path = os.path.join(path, self.dataset_name)
        if not os.path.exists(path):
            print(f"Dataset {self.dataset_name} does not exist. Generating dataset ...")
            if "ConcentricSphere" in self.dataset_name:
                dim = int(re.search(r"\d+", self.dataset_name).group())
                generate_sphere_dataset(dim=dim)
            elif self.dataset_name == "ThreeMinima":
                generate_three_minima_dataset()
            else:
                raise NotImplementedError(f"Dataset with name {self.dataset_name} does not exist!")
            print("Finished dataset generation")

    def return_dataloader(
        self,
        split: str,
        phase: str,
        batch_size: int,
        serial_batches: bool,
        num_threads: int,
        path_to_data=os.path.join(os.path.dirname(__file__), "datasets"),
    ) -> data.DataLoader:
        self.create_if_not_exist(path_to_data)
        if split == "train":
            dataset = self.load_data_train(path_to_data)
            dataloader = data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=not serial_batches,
                num_workers=num_threads,
                drop_last=True,
            )
        else:
            dataset = self.load_data_test(path_to_data)
            dataloader = data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=not serial_batches,
                num_workers=num_threads,
                drop_last=False,
            )
        return dataloader
