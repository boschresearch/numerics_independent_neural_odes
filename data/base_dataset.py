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

import typing

from torch.utils import data as data


class BaseDataset:

    @staticmethod
    def get_properties() -> typing.Dict[str, int]:
        """Specifies 'image_size', 'num_ip_ch', 'num_classes' as dictionary"""
        raise NotImplementedError

    def return_dataloader(self, split: str, phase: str, batch_size: int, serial_batches: bool, num_threads: int,
                          path_to_data: str = 'data/datasets/') -> data.DataLoader:
        raise NotImplementedError
