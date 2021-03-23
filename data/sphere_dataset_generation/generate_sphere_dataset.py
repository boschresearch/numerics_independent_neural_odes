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

from data.sphere_dataset_generation.random_point_in_sphere import random_point_in_sphere

dirname = os.path.dirname(os.path.dirname(__file__))


def generate_sphere(dim, r_min, r_max, n_points):
    sphere_data = None
    for i in range(n_points):
        data_point = random_point_in_sphere(dim=dim, min_radius=r_min, max_radius=r_max)
        if sphere_data is None:
            sphere_data = data_point.reshape((1, 1, dim))
        else:
            sphere_data = torch.cat(
                (sphere_data, data_point.reshape((1, 1, dim))), dim=0
            )
    return sphere_data


def generate_dataset(dim, N):
    r_min = 0.1
    r_max = 0.15
    n_points = int(N / 4)
    label = 0
    data1 = generate_sphere(dim=dim, r_min=r_min, r_max=r_max, n_points=n_points)
    labels1 = torch.ones(n_points, dtype=torch.int64) * label
    r_min = 0.25
    r_max = 0.3
    n_points = int(N / 2)
    label = 1
    data2 = generate_sphere(dim=dim, r_min=r_min, r_max=r_max, n_points=n_points)
    labels2 = torch.ones(n_points, dtype=torch.int64) * label
    r_min = 0.4
    r_max = 0.45
    n_points = int(N / 4)
    label = 0
    data3 = generate_sphere(dim=dim, r_min=r_min, r_max=r_max, n_points=n_points)
    labels3 = torch.ones(n_points, dtype=torch.int64) * label
    data_set = torch.cat((data1, data2, data3))
    labels = torch.cat((labels1, labels2, labels3))
    return data_set, labels


def generate_and_save_data(dim):
    data_dir = os.path.join(dirname, "datasets", f"ConcentricSphere{dim}")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    train_data, train_labels = generate_dataset(dim=dim, N=2000)
    train_set = {"data": train_data, "labels": train_labels}
    file_dir = os.path.join(data_dir, "train_data.pt")
    torch.save(train_set, open(file_dir, "wb"))
    test_data, test_labels = generate_dataset(dim=dim, N=1000)
    train_set = {"data": test_data, "labels": test_labels}
    file_dir = os.path.join(data_dir, "test_data.pt")
    torch.save(train_set, open(file_dir, "wb"))


if __name__ == "__main__":
    # generate Concentric Sphere 2 data set
    generate_and_save_data(dim=2)
