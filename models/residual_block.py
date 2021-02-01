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

import torch
import torch.nn as nn
from .misc import get_activation


class SimpleResidualBlock(nn.Module):
    def __init__(self, in_ch: int, num_filters: int, act='relu'):
        super(SimpleResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, num_filters,
                               kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(num_filters, num_filters,
                               kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(num_filters, in_ch,
                               kernel_size=1, stride=1, padding=0)

        self.activation_function = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        x = self.activation_function(x)
        x = self.conv3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, num_filters: int, act='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, num_filters,
                               kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(num_filters, num_filters,
                               kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, in_ch,
                               kernel_size=1, stride=1, padding=0)

        self.activation_function = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        x = self.activation_function(x)
        x = self.conv3(x)
        return x


class LinearResidualBlock(nn.Module):
    def __init__(self, in_ch: int, num_filters: int, act='relu'):
        super(LinearResidualBlock, self).__init__()
        self.lin1 = nn.Linear(in_ch, num_filters)
        self.lin2 = nn.Linear(num_filters, num_filters)
        self.lin3 = nn.Linear(num_filters, in_ch)

        self.activation_function = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.activation_function(x)
        x = self.lin2(x)
        x = self.activation_function(x)
        x = self.lin3(x)
        return x
