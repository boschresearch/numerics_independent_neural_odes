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

from typing import Dict

import numpy as np
import torch

from options.experiment_options import ExperimentOptions
from trainer.trainer import ModelTrainer
from util.model_evaluation import calculate_accuracy


def find_test_model(solver: str, tol: float) -> (str, float):
    """
    Given the train solver and train tolerance this function returns a test solver and test tolerance.
    See Algorithm 3 in the paper for more details
    :param solver: solver used for training
    :param tol: tolerance used for training
    :return: test solver, test tolerance
    """
    if solver == 'fehlberg2':
        test_solver = 'dopri5'
        tol = tol / 5
        return test_solver, tol
    if solver == 'dopri5':
        test_solver = 'dopri5'
        tol = tol / 10
        return test_solver, tol

    raise NotImplementedError(f"A test solver for {solver} as train solver was not implemented")


def adapt_tol(trainer: ModelTrainer, train_solver_acc: float, train_solver_nfe_dict: Dict[str, torch.Tensor],
              current_iter: int,
              x: torch.Tensor, y: torch.Tensor, opts: ExperimentOptions):
    """
    Adapt the tolerance used for training as described in Algorithm 3 in the paper.
    If tolerance is too large to guarantee continuous dynamics, the tolerance used for training is decreased.
    Else the tolerance is increased, to achieve minimal training time.
    :param trainer: model trainer
    :param train_solver_acc: accuracy reached by the train solver
    :param x: batch data
    :param y: batch labels
    :param opts: experiment options used for training the Neural ODE
    """
    train_solver_nfe = int(train_solver_nfe_dict["nfe_f"][-1].detach())
    threshold = opts.threshold
    test_solver_acc = test_model(trainer=trainer, x=x, y=y, opts=opts, train_solver_nfe=train_solver_nfe)
    dif = np.abs(test_solver_acc - train_solver_acc)
    if dif > threshold:
        tol = trainer.model.feature_ex_block.tol
        new_tol = 0.5 * tol
        trainer.model.feature_ex_block.tol = new_tol
    else:
        tol = trainer.model.feature_ex_block.tol
        new_tol = 1.1 * tol
        trainer.model.feature_ex_block.tol = new_tol
        logits = trainer.forward_one_step(x)
        acc = calculate_accuracy(y=y, logits=logits, batch_size=opts.batch_size,
                                 num_classes=opts.num_classes)

        dif = np.abs(test_solver_acc - acc)

        if dif > threshold:
            new_tol = tol
        elif (trainer.model.feature_ex_block.nfe == train_solver_nfe_dict['nfe_f'][current_iter-5:current_iter]).all()\
                and current_iter > 4:
            """ Do not change tolerance if the number of steps stays constant"""
            new_tol = tol
        trainer.model.feature_ex_block.tol = new_tol


def test_model(trainer: ModelTrainer, x: torch.Tensor, y: torch.Tensor, opts: ExperimentOptions,
               train_solver_nfe: int) -> float:
    """
    Evaluate the test solver accuracy with a test solver and a test tolerance
    :param trainer: model trainer
    :param x: batch data
    :param y: batch labels
    :param opts: experiment options used for training the Neural ODE
    :return: test solver accuracy
    """
    train_tol = trainer.model.feature_ex_block.tol
    train_solver = trainer.model.feature_ex_block.solver
    test_solver, test_tol = find_test_model(solver=train_solver,
                                            tol=train_tol)
    while True:
        trainer.model.feature_ex_block.tol = test_tol
        trainer.model.feature_ex_block.solver = test_solver
        trainer.model.feature_ex_block.nfe = 0
        with torch.no_grad():
            logits = trainer.forward_one_step(x)
        acc = calculate_accuracy(y=y, logits=logits, batch_size=opts.batch_size,
                                 num_classes=opts.num_classes)
        test_solver_nfe = trainer.model.feature_ex_block.nfe
        trainer.model.feature_ex_block.nfe = 0
        if test_solver_nfe > train_solver_nfe:
            break
        """If test solver takes same number of steps as train solver decrease tolerance further if same solver for 
        training and testing is used."""
        test_tol = test_tol / 10

    trainer.model.feature_ex_block.tol = train_tol
    trainer.model.feature_ex_block.solver = train_solver
    return acc
