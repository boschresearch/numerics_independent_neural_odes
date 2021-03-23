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
import pickle
import shutil
from argparse import Namespace
from typing import Type, List, Dict

import torch

import util.helper_functions as utils
from data.base_dataset import BaseDataset
from data.find_dataset_class import find_dataset_class
from options.yaml_service import load_yaml_values


def merge_dicts(opts_dict: Dict, parser_opts: Namespace) -> Dict:
    parser_dict = parser_opts.__dict__
    parser_dict.pop('options_file')
    for key, value in parser_dict.items():
        if value is not None:
            opts_dict[key] = value
    return opts_dict


class ExperimentOptions:
    def __init__(self, parser_opts: Namespace):
        opts_dict = load_yaml_values(parser_opts.options_file)
        opts_dict = merge_dicts(opts_dict, parser_opts)
        self.name = opts_dict['name']  # type: str
        self.dataset = opts_dict['dataset']  # type: str
        self.network = opts_dict['network']  # type: str
        self.adjoint = opts_dict['adjoint']  # type: bool
        self.autodif = opts_dict['autodif']  # type: str
        if self.adjoint:
            self.autodif = 'adjoint'

        self.tol = opts_dict['tol']  # type: float
        self.step_size = opts_dict['step_size']  # type: float
        self.solver = opts_dict['solver']  # type: str
        self.split = opts_dict['split']  # type: str
        self.phase = opts_dict['phase']  # type: str
        self.ch_residual_block = opts_dict['ch_residual_block']  # type: int
        self.act = opts_dict['act']  # type: str
        self.init_type = opts_dict['init_type']  # type: str
        self.init_variance = opts_dict['init_variance']  # type: float
        self.output_dir = opts_dict['output_dir']  # type: str
        self.niter = opts_dict['niter']  # type: int
        self.batch_size = opts_dict['batch_size']  # type: int
        self.serial_batches = opts_dict['serial_batches']  # type: bool
        self.lr = opts_dict['lr']  # type: float
        self.optimizer = opts_dict['optimizer']  # type: str
        self.use_gpu = opts_dict['use_gpu']  # type: bool
        self.gpu_ids = opts_dict['gpu_ids']  # type: str
        self.num_threads = opts_dict['num_threads']  # type: int

        self.model_checkpoint_freq = opts_dict['model_checkpoint_freq']  # type: int
        self.random_seed = opts_dict['random_seed']  # type: int
        self.use_tensorboard = opts_dict['use_tensorboard']  # type: bool
        self.evaluate_test_acc = opts_dict['evaluate_test_acc']  # type: bool

        self.use_adaption_algo = opts_dict['use_step_adaption_algo']  # type: bool
        self.threshold = opts_dict['threshold']  # type: float
        self.initial_tol = opts_dict['initial_tol'] # type: float
        self.adaption_interval = opts_dict['adaption_interval']  # type: int
        self.max_steps = opts_dict['max_steps']  # type: int

        self.evaluate_with_dif_solver = opts_dict['evaluate_with_dif_solver']  # type: bool
        self.test_factor_list = opts_dict['test_factor_list']  # type: List[float]
        self.test_solver_list = opts_dict['test_solver_list']  # type: List[str]

        dataset_class = find_dataset_class(self.dataset)  # type: Type[BaseDataset]
        dataset_properties = dataset_class.get_properties()
        self.image_size = dataset_properties["image_size"]
        self.num_ip_ch = dataset_properties["num_ip_ch"]
        self.num_classes = dataset_properties["num_classes"]
        self.experiment_dir = os.path.join(self.output_dir, self.name)
        self.checkpoints_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.fixed_step_solver_list = ['euler', 'rk4', 'midpoint']
        self.fixed_step_solver = self.solver in self.fixed_step_solver_list

        if self.use_tensorboard:
            self.tensorboard_dir = os.path.join(
                self.experiment_dir, 'tensorboard_logs')

        self.initialize_setup()

    def initialize_setup(self):
        self.experiment_dir = os.path.abspath(self.experiment_dir)
        if os.path.exists(self.experiment_dir):
            if os.listdir(self.experiment_dir):
                print(f"The experiment directory {self.experiment_dir} provided by the user is not empty. "
                      f"If you continue the contents of this folder will be deleted!\n"
                      f"If you want to change the folder please "
                      "restart the code using --experiment_dir command line option.")
                input("Press ENTER to continue (deletes experiment directory!) ...")
                shutil.rmtree(self.experiment_dir)

        utils.mkdirs(self.experiment_dir)
        utils.mkdirs(self.checkpoints_dir)
        if self.use_tensorboard:
            utils.mkdirs(self.tensorboard_dir)
        self._set_gpu_options()
        self.save_options()

    def _set_gpu_options(self):
        # set gpu ids
        if self.use_gpu:
            str_gpu_ids = self.gpu_ids.split(',')
            self.gpu_ids = []
            for str_id in str_gpu_ids:
                id = int(str_id)
                if id >= 0:
                    self.gpu_ids.append(id)
            if len(self.gpu_ids) > 0:
                torch.cuda.set_device(self.gpu_ids[0])

            # Total number of gpus
            num_gpus = len(self.gpu_ids)

            # Update the batch size
            self.batch_size = num_gpus * self.batch_size
        else:
            self.num_threads = 0

    def save_options(self):
        file_path = os.path.join(self.experiment_dir, 'options')
        utils.mkdirs(file_path)
        file_name = os.path.join(file_path, 'opts')
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(self).items()):
                opt_file.write('{:>25}: {:<30}\n'.format(str(k), str(v)))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(self, opt_file)
