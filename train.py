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

import argparse
import copy
import os
from typing import Generator

import torch

# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import data
import data.create_dataloader
from evaluate_with_dif_solver import evaluate_with_dif_solver
from options.experiment_options import ExperimentOptions
from options.initialize import initialize
from trainer.trainer import ModelTrainer
from util.helper_functions import inf_generator, return_order
from util.model_evaluation import calculate_accuracy, evaluate_model
from util.plot_results import plot_results
from util.step_adaption_algo import find_initial_step_size, adapt_step_size
from util.tol_adaption_algo import adapt_tol


class TrainModel:
    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser = initialize(parser)
        opts, unknown = parser.parse_known_args()
        self.opts = ExperimentOptions(opts)
        self.data_generator = self._get_data_generator()
        self.test_dataloader = self._get_test_dataloader()

        self.train_acc = None
        self.test_acc = None
        self.nfe_f = None
        self.nfe_b = None
        self.loss = None

        self.acc_log = {
            "train": torch.empty(self.opts.niter),
            "test": torch.empty(self.opts.niter),
        }
        self.loss_log = torch.empty(self.opts.niter)
        self.nfe_log = {
            "nfe_f": torch.empty(self.opts.niter),
            "nfe_b": torch.empty(self.opts.niter),
        }

        self.trainer = ModelTrainer(self.opts)
        if self.opts.use_gpu:
            self.trainer.model.cuda()

        # By default the device is cpu
        self.device = torch.device("cpu")
        if self.opts.use_gpu:
            self.device = torch.device("cuda:" + str(self.opts.gpu_ids[0]))

        # Initialize the summary writer

        if self.opts.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.opts.tensorboard_dir)

    def run(self):
        torch.cuda.empty_cache()
        # Set random seed
        torch.manual_seed(self.opts.random_seed)

        loss_function = torch.nn.CrossEntropyLoss().to(self.device)
        # Time input for the ODE
        t = torch.as_tensor([0.0, 1.0]).to(self.device)

        print("Starting training....")
        if self.opts.use_adaption_algo:
            self._initialize_adaption_algo()
        for current_iter in range(self.opts.niter):
            self._iterate_one_training_step(current_iter, loss_function, t)
        if self.opts.evaluate_with_dif_solver:
            results = evaluate_with_dif_solver(
                trainer=self.trainer,
                test_dataloader=self.test_dataloader,
                opts=self.opts,
                device=self.device,
            )

            torch.save(
                results,
                os.path.join(
                    self.opts.experiment_dir,
                    f"eval_with_dif_solver_iter_{self.opts.niter - 1}.pt",
                ),
            )
        plot_results(self.opts)

    def _get_data_generator(self) -> Generator:
        # load the dataset
        dataloader = data.create_dataloader.create_dataloader(self.opts)
        print(
            "\n{} dataloader of size {} was created\n".format(
                self.opts.dataset.upper(), len(dataloader)
            )
        )
        # Wrap pytorch's dataloader in a generator function
        return inf_generator(dataloader)

    def _get_test_dataloader(self) -> DataLoader:
        test_opts = copy.deepcopy(self.opts)
        test_opts.split = "test"
        return data.create_dataloader.create_dataloader(test_opts)

    def _initialize_adaption_algo(self):
        x, _ = self.data_generator.__next__()
        x = x.to(self.device)
        if self.opts.fixed_step_solver:
            step_size = find_initial_step_size(
                mymodel=self.trainer.model,
                batch_data=x,
                order=return_order(self.opts.solver),
            )
            self.trainer.model.feature_ex_block.options["step_size"] = step_size
        else:
            tol = self.opts.initial_tol
            self.trainer.model.feature_ex_block.tol = tol

    def _iterate_one_training_step(
        self, current_iter: int, loss_function: _Loss, t: torch.Tensor
    ):
        self.trainer.model.train()
        self.trainer.optimizer.zero_grad()

        self.trainer.model.feature_ex_block.nfe = 0
        x, y = self.data_generator.__next__()
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.trainer.forward_one_step(x, t)
        self.loss = loss_function(logits, y)
        self.nfe_f = self.trainer.model.feature_ex_block.nfe
        self.trainer.model.feature_ex_block.nfe = 0
        self.loss.backward()
        self.nfe_b = self.trainer.model.feature_ex_block.nfe
        self.trainer.model.feature_ex_block.nfe = 0

        self.train_acc = calculate_accuracy(
            logits, y, self.opts.num_classes, self.opts.batch_size
        )

        if self.opts.evaluate_test_acc:
            with torch.no_grad():
                self.trainer.model.eval()
                self.test_acc = evaluate_model(
                    self.trainer.model, self.test_dataloader, self.opts, self.device
                )

        self._save_current_state(current_iter)

        if self.opts.use_adaption_algo:
            self._apply_step_adaption_algo(current_iter, self.train_acc, x, y)

        if self.opts.use_tensorboard:
            self._create_tensorboard_logs(current_iter)
        self._print_training_info(current_iter)
        self.trainer.optimizer.step()
        torch.cuda.empty_cache()

    def _save_current_state(self, current_iter: int):
        self.nfe_log["nfe_f"][current_iter] = self.nfe_f
        self.nfe_log["nfe_b"][current_iter] = self.nfe_b
        self.loss_log[current_iter] = self.loss.cpu().detach()
        self.acc_log["train"][current_iter] = self.train_acc
        if self.opts.evaluate_test_acc:
            self.acc_log["test"][current_iter] = self.test_acc

        torch.save(self.loss_log, os.path.join(self.opts.experiment_dir, "loss_log.pt"))
        torch.save(self.acc_log, os.path.join(self.opts.experiment_dir, "acc_log.pt"))
        torch.save(self.nfe_log, os.path.join(self.opts.experiment_dir, "nfe_log.pt"))

        # Save the current model
        if (current_iter + 1) % self.opts.model_checkpoint_freq == 0 or (
            current_iter + 1
        ) == self.opts.niter:
            self.trainer.checkpoint_model_state(current_iter, self.opts.checkpoints_dir)

    def _create_tensorboard_logs(self, current_iter: int):
        self.writer.add_scalar("ACC/train", self.train_acc, current_iter + 1)
        self.writer.add_scalar("NFE/forward", self.nfe_f, current_iter + 1)
        self.writer.add_scalar("NFE/backward", self.nfe_b, current_iter + 1)

    def _print_training_info(self, current_iter: int):
        print_str = "Iter {} \b\b\t NFE-F {:.2f} \t NFE-B {:.2f}" "\t Train Acc {:.3f}%"
        print_vars = (current_iter + 1, self.nfe_f, self.nfe_b, self.train_acc)
        if self.test_acc is not None:
            print_str = print_str + "\t Test Acc {:.3f}%"
            print_vars = print_vars + (self.test_acc,)

        print(
            print_str.format(*print_vars),
            file=open(os.path.join(self.opts.experiment_dir, "output.txt"), "a"),
        )

    def _apply_step_adaption_algo(
        self, current_iter: int, train_acc: float, x: torch.Tensor, y: torch.Tensor,
    ):

        if (current_iter + 1) % self.opts.adaption_interval == 0:
            if self.opts.fixed_step_solver:
                adapt_step_size(
                    trainer=self.trainer,
                    train_solver_acc=train_acc,
                    x=x,
                    y=y,
                    opts=self.opts,
                )
            else:
                adapt_tol(
                    trainer=self.trainer,
                    train_solver_acc=train_acc,
                    x=x,
                    y=y,
                    opts=self.opts,
                    train_solver_nfe_dict=self.nfe_log,
                    current_iter=current_iter,
                )


if __name__ == "__main__":
    TrainModel().run()
