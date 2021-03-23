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


from random import random

import torch

"""
The following function is adapted from Augmented-Neural-ODEs V 1.0
( https://github.com/EmilienDupont/augmented-neural-odes/
Copyright (c) 2019 Emilien Dupont, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
to produce the 2 dimensional concentric sphere dataset.
"""


def random_point_in_sphere(dim, min_radius, max_radius):
    """Returns a point sampled uniformly at random from a sphere if min_radius
    is 0. Else samples a point approximately uniformly on annulus.
    Parameters
    ----------
    dim : int
        Dimension of sphere
    min_radius : float
        Minimum distance of sampled point from origin.
    max_radius : float
        Maximum distance of sampled point from origin.
    """
    # Sample distance of point from origin
    unif = random()
    distance = (max_radius - min_radius) * (unif ** (1.0 / dim)) + min_radius
    # Sample direction of point away from origin
    direction = torch.randn(dim)
    unit_direction = direction / torch.norm(direction, 2)
    return distance * unit_direction
