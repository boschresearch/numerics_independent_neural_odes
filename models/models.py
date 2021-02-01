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
from torch import nn as nn

from options.experiment_options import ExperimentOptions
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint


class ODENet(nn.Module):
    """
    Function parameterized via a neural-network that approximates the ode state
    derivative
    """

    def __init__(self, residualblock: nn.Module):
        super(ODENet, self).__init__()

        self.net = residualblock
        self.nfe = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        x = self.net(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc: ODENet, opts: ExperimentOptions):
        super(ODEBlock, self).__init__()

        self.odefunc = odefunc
        self.tol = opts.tol
        self.solver = opts.solver
        self.autodif = opts.autodif

        self.default_integration_time = torch.tensor([0, 1]).float()

        if opts.autodif == "adjoint":
            self.ode_solver = odeint_adjoint
        else:
            self.ode_solver = odeint
        if opts.fixed_step_solver:
            self.options = {"step_size": opts.step_size}
        else:
            self.options = {}

    def forward(
        self, x: torch.Tensor, t: torch.Tensor = None, save=False
    ) -> torch.Tensor:
        # The time dependency condition is handled in the odefunc method.
        # Pass the intergation time anyway for the odesolver.
        if t is not None:
            self.integration_time = t
        else:
            self.integration_time = self.default_integration_time

        self.integration_time = self.integration_time.type_as(x)

        out = self.ode_solver(
            self.odefunc,
            x,
            self.integration_time,
            rtol=self.tol,
            atol=self.tol,
            method=self.solver,
            options=self.options,
        )

        # Return only the last time point
        return out[-1]

    @property
    def nfe(self) -> int:
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value: int):
        self.odefunc.nfe = value
