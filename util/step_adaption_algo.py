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
import numpy as np

from models.full_model import ModelBase
from options.experiment_options import ExperimentOptions
from trainer.trainer import ModelTrainer
from util.model_evaluation import calculate_accuracy


def find_initial_step_size(mymodel: ModelBase, batch_data: torch.Tensor, order: int) -> float:
    """
    Calculate initial step size as described in Hairer Wanner
    :param mymodel: Model to be trained
    :param batch_data: Current batch of data
    :param order: order of the solver to be used
    :return: suggestion for an initial step size
    """
    with torch.no_grad():
        p = order
        y0 = batch_data
        d0 = torch.mean(torch.norm(y0, dim=-1))
        f_y0 = mymodel.feature_ex_block.odefunc.net(y0)
        d1 = torch.mean(torch.norm(f_y0, dim=-1))
        if d1 < 1e-5 or d0 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1
        y1 = y0 + h0*f_y0
        f_y1 = mymodel.feature_ex_block.odefunc.net(y1)
        d2 = torch.mean(torch.norm(f_y1 - f_y0, dim=-1))/h0
        if torch.max(d1, d2) < 1e-15:
            h1 = torch.max(1e-6, h0*1e-3)
        else:
            h1 = (0.01/torch.max(d1, d2))**(1/(p+1))
        step_size = torch.min(100*h0, h1)
        return step_size.detach().item()


def find_test_model(solver: str, step_size: float) -> (str, float):
    acc_fact = 50
    if solver == 'euler':
        if 1./step_size > acc_fact:
            test_solver = 'midpoint'
        else:
            test_solver = 'midpoint'
            step_size = (step_size/acc_fact)**(1./2)
        return test_solver, step_size
    # Implementation for Euler and rk4. Results are shown in Appendix.
    # if solver == 'euler':
    #     if 1. / step_size ** 3 > acc_fact:
    #         test_solver = 'rk4'
    #     else:
    #         test_solver = 'rk4'
    #         step_size = (step_size / acc_fact) ** (1. / 4)
    #     return test_solver, step_size
    if solver == 'midpoint':  # This is not tested!
        if 1. / step_size ** 2 > acc_fact:
            test_solver = 'rk4'
        else:
            test_solver = 'rk4'
            step_size = (step_size ** 2 / acc_fact) ** (1. / 4)
        return test_solver, step_size

    raise NotImplementedError("A higher order method for this solver was not implemented")


def adapt_step_size(trainer: ModelTrainer, train_solver_acc: float, x: torch.Tensor, y: torch.Tensor,
                    opts: ExperimentOptions):
    threshold = opts.threshold
    max_steps = opts.max_steps
    test_solver_acc = test_model(trainer=trainer, x=x, y=y, opts=opts)
    dif = np.abs(test_solver_acc - train_solver_acc)
    if dif > threshold:
        step_size = trainer.model.feature_ex_block.options['step_size']
        if 1/step_size * 2 > max_steps:
            if int(1/step_size) == max_steps:
                print("WARNING: Cannot increase step size further!")
                new_step_size = step_size
            else:
                new_step_size = 2*step_size/(max_steps*step_size + 1)
        else:
            new_step_size = 0.5 * step_size
        trainer.model.feature_ex_block.options['step_size'] = new_step_size
    else:
        step_size = trainer.model.feature_ex_block.options['step_size']
        new_step_size = 1.1 * step_size
        if new_step_size > 1.0:
            new_step_size = 1.0
        trainer.model.feature_ex_block.options['step_size'] = new_step_size
        logits = trainer.forward_one_step(x)
        acc = calculate_accuracy(y=y, logits=logits, batch_size=opts.batch_size,
                                 num_classes=opts.num_classes)
        dif = np.abs(test_solver_acc - acc)
        if dif > threshold:
            new_step_size = step_size
        trainer.model.feature_ex_block.options['step_size'] = new_step_size


def test_model(trainer: ModelTrainer, x: torch.Tensor, y: torch.Tensor, opts: ExperimentOptions) -> float:
    train_step_size = trainer.model.feature_ex_block.options['step_size']
    train_solver = trainer.model.feature_ex_block.solver
    test_solver, test_step_size = find_test_model(solver=train_solver,
                                                  step_size=train_step_size)

    trainer.model.feature_ex_block.options['step_size'] = test_step_size
    trainer.model.feature_ex_block.solver = test_solver
    with torch.no_grad():
        logits = trainer.forward_one_step(x)
    acc = calculate_accuracy(y=y, logits=logits, batch_size=opts.batch_size,
                             num_classes=opts.num_classes)
    trainer.model.feature_ex_block.options['step_size'] = train_step_size
    trainer.model.feature_ex_block.solver = train_solver
    return acc
