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

import importlib
from typing import Type

from data.base_dataset import BaseDataset


def find_dataset_class(dataset_name: str) -> Type[BaseDataset]:
    filename = 'data.' + dataset_name + '_dataset'
    datasetlib = importlib.import_module(filename)

    dataset_class = None
    target_class_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_class_name.lower():
            dataset_class = cls

    if dataset_class is None:
        raise ValueError('Could not find {} dataset'.format(dataset_name))

    return dataset_class
