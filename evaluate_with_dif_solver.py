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

from typing import Union, Dict

import torch
import os

from torch.utils.data import DataLoader

import data.create_dataloader
from options.experiment_options import ExperimentOptions
from trainer.trainer import ModelTrainer
from util.model_evaluation import evaluate_model
import copy
import data
import pickle


def load_model(path: str, model_iter: int, use_gpu=True) -> (ModelTrainer, DataLoader, ExperimentOptions):
    opts_folder = 'options'
    file = 'opts.pkl'
    file_path = os.path.join(path, opts_folder, file)

    with open(file_path, "rb") as input_file:
        opts = pickle.load(input_file)
    opts.fixed_step_solver = True
    if not use_gpu:
        opts.use_gpu = False
    check_folder = 'checkpoints'

    test_opts = copy.deepcopy(opts)
    test_opts.split = 'test'
    test_dataloader = data.create_dataloader.create_dataloader(test_opts)

    file = f'model_iter_{model_iter}.pth'
    file_path = os.path.join(path, check_folder, file)

    trainer = ModelTrainer(opts)
    if use_gpu:
        state_dict = torch.load(file_path)
    else:
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    trainer.model.load_state_dict(state_dict['model_state'])
    return trainer, test_dataloader, opts


def evaluate_with_dif_solver(trainer: ModelTrainer, opts: ExperimentOptions, test_dataloader: DataLoader,
                             device: Union[torch.device, str]) -> Dict[str, Dict[float, float]]:
    results = dict()
    with torch.no_grad():
        for test_solver in opts.test_solver_list:
            results[test_solver] = dict()
            for test_factor in opts.test_factor_list:
                if test_solver in opts.fixed_step_solver_list:
                    trainer.model.feature_ex_block.options['step_size'] = 1 / (opts.step_size * test_factor)
                    trainer.model.feature_ex_block.solver = test_solver
                    results[test_solver][test_factor] = evaluate_model(trainer.model,
                                                                       test_dataloader=test_dataloader,
                                                                       opts=opts,
                                                                       device=device)
                else:
                    trainer.model.feature_ex_block.tol = opts.tol * test_factor
                    trainer.model.feature_ex_block.solver = test_solver
                    results[test_solver][test_factor] = evaluate_model(trainer.model,
                                                                       test_dataloader=test_dataloader,
                                                                       opts=opts,
                                                                       device=device)
    reset_model_parameters(opts, trainer)
    return results


def reset_model_parameters(opts: ExperimentOptions, trainer: ModelTrainer):
    if opts.fixed_step_solver:
        trainer.model.feature_ex_block.options['step_size'] = 1 / opts.step_size
    else:
        trainer.model.feature_ex_block.tol = opts.tol
    trainer.model.feature_ex_block.solver = opts.solver


def main():
    """
    Evaluates existing model at checkpoint model_iter
    Default: Use evaluation configuration from opts.pkl.
    To change certain evaluation parameters e.g. test_factor_list, test_solver_list set them via the opts variable.
    E.g. opts.test_factor_list =  [0.1, 0.2, 3]
    """
    path = os.path.join('experiments', 'neural_ode')
    model_iter = 0
    use_gpu = False
    device = torch.device('cpu')
    if use_gpu:
        device = torch.device('cuda')
    trainer, test_dataloader, opts = load_model(path, model_iter=model_iter, use_gpu=use_gpu)
    trainer.model.to(device)

    eval_file_path = os.path.join(opts.experiment_dir, f'eval_with_dif_solver_iter_{model_iter}.pt')
    if os.path.exists(eval_file_path):
        input(f"File {eval_file_path} already exists! Press Enter to overwrite.")
    results = evaluate_with_dif_solver(trainer, opts, test_dataloader, device)
    torch.save(results, eval_file_path)


if __name__ == "__main__":
    main()
