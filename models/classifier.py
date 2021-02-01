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
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    def __init__(self, num_hidden_units: int, num_classes: int):
        super(SimpleClassifier, self).__init__()
        self.lin1 = nn.Linear(num_hidden_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        return x
