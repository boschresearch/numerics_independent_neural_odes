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
from typing import Dict

import yaml

dirname = os.path.dirname(__file__)
experiment_dir = os.path.join(dirname, "..", "experiments")


def load_yaml(path: str = os.path.join(dirname, "default_config.yaml")) -> Dict:
    with open(path, "r") as stream:
        result = yaml.safe_load(stream)
        return result if result is not None else dict()


def load_yaml_values(path: str = os.path.join(dirname, "default_config.yaml")) -> Dict:
    simplified_dict = dict()
    exp_dict = load_yaml(path)
    for settings_key, settings_dict in exp_dict.items():
        simplified_dict[settings_key] = settings_dict["value"]
    return simplified_dict
