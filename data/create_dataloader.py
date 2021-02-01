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

from torch.utils.data import DataLoader

from data.find_dataset_class import find_dataset_class
from options.experiment_options import ExperimentOptions


def create_dataloader(opts: ExperimentOptions) -> DataLoader:
    dataset = find_dataset_class(opts.dataset)
    dataset_instance = dataset()
    dataloader = dataset_instance.return_dataloader(opts.split, opts.phase, opts.batch_size, opts.serial_batches,
                                                    opts.num_threads)
    return dataloader
