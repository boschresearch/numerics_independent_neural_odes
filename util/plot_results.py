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

import matplotlib.pyplot as plt
import numpy
import torch
import matplotlib.pylab as pl

from options.experiment_options import ExperimentOptions


def plot_results(opts: ExperimentOptions):
    print("plotting data...")
    experiment_dir = opts.experiment_dir
    figure_path = os.path.join(experiment_dir, "figures")
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    plot_nfe_f(experiment_dir, figure_path)
    plot_train_accuracy(experiment_dir, figure_path)
    if opts.evaluate_test_acc:
        plot_test_accuracy(experiment_dir, figure_path)
    plot_loss(experiment_dir, figure_path)
    if opts.evaluate_with_dif_solver:
        plot_evaluate_dif_solver(experiment_dir, figure_path, opts)
    print(f"Finished plotting - you can find the result here:\n {os.path.abspath(figure_path)}")


def plot_nfe_f(path, fig_path):
    fig, ax = plt.subplots()
    nfe = torch.load(os.path.join(path, "nfe_log.pt"))
    plt.plot(nfe["nfe_f"], lw=0.5, alpha=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("#nfe")
    plt.savefig(os.path.join(fig_path, f"nfe.png"), dpi=300)
    plt.clf()
    plt.close(fig=fig)


def plot_test_accuracy(path, fig_path):
    fig, ax = plt.subplots()
    acc = torch.load(os.path.join(path, "acc_log.pt"))
    plt.plot(acc["test"], lw=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test Accurracy")
    ax.set_ylim([0.0, 1.0])
    plt.savefig(os.path.join(fig_path, f"test_acc.png"), dpi=300)
    plt.clf()
    plt.close(fig=fig)


def plot_train_accuracy(path, fig_path):
    fig, ax = plt.subplots()
    acc = torch.load(os.path.join(path, "acc_log.pt"))
    plt.plot(acc["train"], lw=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Train Accurracy")
    ax.set_ylim([0.0, 1.1])
    plt.savefig(os.path.join(fig_path, f"train_acc.png"), dpi=300)
    plt.clf()
    plt.close(fig=fig)


def plot_loss(path, fig_path):
    fig, ax = plt.subplots()
    loss = torch.load(os.path.join(path, "loss_log.pt"))
    plt.plot(loss, lw=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    plt.savefig(os.path.join(fig_path, f"loss.png"), dpi=300)
    plt.clf()
    plt.close(fig=fig)


def plot_evaluate_dif_solver(path, fig_path, opts: ExperimentOptions):
    colors = pl.cm.tab10(numpy.linspace(0.0, 1.0, 10))
    fig, ax = plt.subplots()
    result = torch.load(os.path.join(path, f"eval_with_dif_solver_iter_{opts.niter - 1}.pt"))
    print(result)
    i = 0
    for solver in opts.test_solver_list:
        x_values = numpy.array(list(result[solver].keys()))
        if opts.fixed_step_solver:
            x_values = x_values/opts.step_size
        else:
            x_values = x_values/opts.tol
        accuracy = result[solver].values()
        plt.semilogx(
            x_values,
            accuracy,
            ".-",
            label=solver.replace("_", " "),
            alpha=0.5,
            color=colors[i]
        )
        i += 1
    if opts.fixed_step_solver:
        plt.semilogx(1/opts.step_size, result[opts.solver][1.0], ".", color=colors[0], label="test config=train config")
    else:
        plt.semilogx(1/opts.tol, result[opts.solver][1.0], ".", color=colors[0], label="test config=train config")
    ax.set_xlabel("1/tol")
    if opts.fixed_step_solver:
        ax.set_xlabel("#Steps")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim([0.0, 1.1])
    if opts.dataset == "cifar10":
        ax.set_ylim([0.0, 0.7])
    plt.legend()
    plt.savefig(os.path.join(fig_path, f"eval_dif_solver.png"), dpi=300)
    plt.clf()
    plt.close(fig=fig)



