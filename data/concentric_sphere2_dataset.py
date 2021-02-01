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

from .toy_dataset_base import ToyDataset


class ConcentricSphere2Dataset(ToyDataset):
    def __init__(self):
        super(ConcentricSphere2Dataset, self).__init__()
        self.dataset_name = 'ConcentricSphere2'
        self.number_of_classes = 3

    @staticmethod
    def get_properties():
        return {
            'image_size': 2,
            'num_ip_ch': 1,
            'num_classes': 2,
        }
