# -*- coding: utf-8 -*-
# Copyright (c) 2022 Robert Bosch GmbH
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

import torch

from data.three_minima_dataset_generation.energy_landscape import ODEModel, ThreeMinima

dirname = os.path.dirname(os.path.dirname(__file__))


def generate_dataset(N: int):
    model = ThreeMinima
    x_range = [-1, 2]
    y_range = [-2, 2]
    gamma = 0.2
    step_size = 0.02
    solver = "rk4"
    tf = 100
    x_data, v_data = sample_start_values(N, x_range, y_range)
    data = torch.stack((x_data, v_data), dim=0).transpose(0, 1)
    mymodel = ODEModel(model, None, step_size=step_size, solver=solver, gamma=gamma)
    result = mymodel.solve(data, tf)
    x = result[:, 0]
    a = (torch.abs(result[:, 1]) < 1e-3).all().item()
    if not a:
        raise Exception(
            "Model has not converged - the final velocity is too large. Increase number of steps"
        )
    category = 0
    for m in mymodel.func.true_minima:
        x[torch.abs(x - m) < 0.02] = category
        category += 1
    y_data = x.type(torch.long)
    return data.unsqueeze(1), y_data


def sample_start_values(length, x_range, y_range):
    x_data = torch.distributions.Uniform(x_range[0], x_range[1]).sample([length])
    v_data = torch.distributions.Uniform(y_range[0], y_range[1]).sample([length])
    return x_data, v_data


def generate_three_minima_dataset():
    data_dir = os.path.join(dirname, "datasets", f"ThreeMinima")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    train_data, train_labels = generate_dataset(N=2000)
    train_set = {"data": train_data, "labels": train_labels}
    file_dir = os.path.join(data_dir, "train_data.pt")
    plt.scatter(train_data[:, 0, 0], train_data[:, 0, 1], c=train_labels)
    plt.savefig(os.path.join(data_dir, "train_data.png"), dpi=500)
    torch.save(train_set, open(file_dir, "wb"))
    test_data, test_labels = generate_dataset(N=1000)
    train_set = {"data": test_data, "labels": test_labels}
    file_dir = os.path.join(data_dir, "test_data.pt")
    torch.save(train_set, open(file_dir, "wb"))


if __name__ == "__main__":
    generate_three_minima_dataset()
