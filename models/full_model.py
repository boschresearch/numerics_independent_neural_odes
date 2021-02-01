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

import torch
import torch.nn as nn
from torch.nn import init

from options.experiment_options import ExperimentOptions
from .residual_block import LinearResidualBlock, \
    ResidualBlock, SimpleResidualBlock
from .models import ODEBlock, ODENet
from .classifier import SimpleClassifier
from .misc import Flatten


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m: nn.Module):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find(
                    'Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented'
                        % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class ODENetModelSimple(ModelBase):
    def __init__(self, opts: ExperimentOptions):
        super(ODENetModelSimple, self).__init__()

        # Feature extractor block
        self.feature_ex_block = ODEBlock(
            ODENet(SimpleResidualBlock(opts.num_ip_ch,
                                       opts.ch_residual_block,
                                       act=opts.act)), opts)

        # Classifier block
        self.flatten = Flatten()
        self.classifier = SimpleClassifier(num_hidden_units=opts.image_size,
                                           num_classes=opts.num_classes,
                                           )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        out = self.feature_ex_block(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out


class ODENetModelLinear(ModelBase):
    def __init__(self, opts: ExperimentOptions):
        super(ODENetModelLinear, self).__init__()

        # Feature extractor block
        self.feature_ex_block = ODEBlock(
            ODENet(LinearResidualBlock(opts.image_size,
                                       opts.ch_residual_block,
                                       act=opts.act)), opts)

        # Classifier block
        self.flatten = Flatten()
        self.classifier = SimpleClassifier(num_hidden_units=opts.image_size,
                                           num_classes=opts.num_classes,
                                           )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        out = self.feature_ex_block(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out


class ODENetModel(ModelBase):
    def __init__(self, opts: ExperimentOptions):
        super(ODENetModel, self).__init__()

        # Feature extractor block
        self.feature_ex_block = ODEBlock(
            ODENet(ResidualBlock(opts.num_ip_ch,
                                 opts.ch_residual_block,
                                 act=opts.act)), opts)

        # Classifier block
        self.flatten = Flatten()
        self.classifier = SimpleClassifier(num_hidden_units=opts.image_size ** 2 * opts.num_ip_ch,
                                           num_classes=opts.num_classes,
                                           )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        out = self.feature_ex_block(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out
