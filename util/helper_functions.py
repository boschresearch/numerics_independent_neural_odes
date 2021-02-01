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
from typing import List, Union, Iterable, Generator, Dict

import numpy as np


def mkdirs(paths: Union[List[str], str]):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def inf_generator(iterable: Iterable) -> Generator:
    """Allows training with DataLoaders in a single infinite loop:
       Usage: for i, (x, y) in enumerate(inf_generator(train_loader))
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def one_hot(x: np.core.multiarray, K: int) -> np.core.multiarray:
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def return_order(solver: str) -> Union[int, Dict[str, int]]:
    """ For adaptive methods this also returns the number of function evaluations per step. This should at some point
    be directly accessible via the solver property."""
    if solver == 'euler':
        return 1
    if solver == 'midpoint':
        return 2
    if solver == 'rk4':
        return 4
    if solver == 'fehlberg2':
        return {'sol': 2, 'error': 1, 'nfe_step': 3}
    if solver == 'heun_euler':
        return {'sol': 2, 'error': 1, 'nfe_step': 2}
    if solver == 'bogacki_shampine':
        return {'sol': 4, 'error': 3, 'nfe_step': 4}
    if solver == 'dopri5_clip' or solver == 'dopri5':
        return {'sol': 4, 'error': 5, 'nfe_step': 7}
    raise NotImplementedError(f"Order not implemented for solver {solver}")
