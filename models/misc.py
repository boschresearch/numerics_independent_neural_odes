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

from typing import Union

import torch
import torch.nn as nn


def get_activation(act='relu') -> Union[nn.ReLU, nn.LeakyReLU, nn.Sigmoid]:
    if act == 'relu':
        return nn.ReLU(inplace=True)
    if act == 'lrelu':
        return nn.LeakyReLU()
    if act == 'sigmoid':
        return nn.Sigmoid()
    raise NotImplementedError('Activation function {} not implemented'.format(act))


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
