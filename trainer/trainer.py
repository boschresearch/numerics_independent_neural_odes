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

import torch
import torch.nn as nn
import torch.optim as optim

from models.full_model import ODENetModelSimple, ODENetModel, ODENetModelLinear, ModelBase
from options.experiment_options import ExperimentOptions


class ModelTrainer:
    """
    Trainer creates the complete model and its optimizer. 
    """

    def __init__(self, opts: ExperimentOptions):
        super(ModelTrainer, self).__init__()

        self.opts = opts
        self.model = self.create_model(opts)
        self.model.init_weights(init_type=opts.init_type,
                                gain=opts.init_variance)
        self.optimizer = self.init_optimizer(opts)

    def create_model(self, opts: ExperimentOptions) -> ModelBase:
        if opts.network == 'simple_odenet':
            return ODENetModelSimple(opts)
        if opts.network == 'linear_odenet':
            return ODENetModelLinear(opts)
        if opts.network == 'odenet':
            return ODENetModel(opts)
        raise NotImplementedError(f'Model {opts.network} not implemented')

    def init_optimizer(self, opts: ExperimentOptions) -> optim.Optimizer:
        if opts.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=opts.lr)
        if opts.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=opts.lr)
        if opts.optimizer == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=opts.lr)
        raise NotImplementedError(f'Optimizer {opts.optimizer} not implemented')

    def forward_one_step(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        if self.opts.use_gpu:
            model_inputs = (x, t)
            out = nn.parallel.data_parallel(self.model, model_inputs,
                                            self.opts.gpu_ids)
        else:
            out = self.model(x, t)
        return out

    def checkpoint_model_state(self, current_iter: int, checkpoints_dir: str):
        torch.save({
            'iter': current_iter,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, os.path.join(checkpoints_dir,
                        'model_iter_' + str(current_iter) + '.pth'))
