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

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.full_model import ModelBase
from options.experiment_options import ExperimentOptions
from util.helper_functions import one_hot


def calculate_accuracy(
    logits: torch.Tensor, y: torch.Tensor, num_classes: int, batch_size: int
) -> float:
    y = one_hot(np.array(y.detach().cpu().numpy()), num_classes)
    target_class = np.argmax(y, axis=1)
    predicted_class = np.argmax(logits.detach().cpu().numpy(), axis=1)
    return np.sum(predicted_class == target_class) / batch_size


def evaluate_model(
    mymodel: ModelBase,
    test_dataloader: DataLoader,
    opts: ExperimentOptions,
    device: Union[torch.device, str] = "cpu",
) -> float:
    counter = 0
    test_acc = 0.0
    with torch.no_grad():
        for data, labels in test_dataloader:
            counter += 1
            data = data.to(device)
            outputs = mymodel(data)
            y = labels
            correct_predictions = calculate_accuracy(
                outputs, y, num_classes=opts.num_classes, batch_size=opts.batch_size
            )
            test_acc += correct_predictions
    return test_acc / counter
