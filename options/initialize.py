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
from argparse import ArgumentParser
from pydoc import locate
from typing import Dict

from options.yaml_service import load_yaml

dirname = os.path.dirname(__file__)
experiment_dir = os.path.join(dirname, '..', 'experiments')


def initialize(parser: ArgumentParser) -> ArgumentParser:
    # Experiment options
    parser.add_argument('--options_file', type=str,
                        default=os.path.join(dirname, '..', 'options', 'default_config.yaml'),
                        help='File to use to load options from')
    config_dict = load_yaml(os.path.join(dirname, '..', 'options', 'default_config.yaml'))
    parser = initialize_parser_from_dict(config_dict, parser)

    return parser


def initialize_parser_from_dict(config_dict: Dict, parser: ArgumentParser) -> ArgumentParser:
    for key, settings in config_dict.items():
        if 'choices' in settings:
            if 'nargs' in settings:
                parser.add_argument(f'--{key}', type=locate(settings['type']),
                                    nargs=settings['nargs'],
                                    choices=settings['choices'],
                                    help=settings['help'])
            else:
                parser.add_argument(f'--{key}', type=locate(settings['type']),
                                    choices=settings['choices'],
                                    help=settings['help'])
        elif 'nargs' in settings:
            parser.add_argument(f'--{key}', type=locate(settings['type']),
                                nargs=settings['nargs'],
                                help=settings['help'])
        else:

            parser.add_argument(f'--{key}', type=locate(settings['type']),
                                help=settings['help'])
    return parser
