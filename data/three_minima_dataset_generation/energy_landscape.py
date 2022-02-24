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

import torch
from torchdiffeq import odeint_adjoint, odeint


class EnergyLandscape(torch.nn.Module):
    def __init__(self, params=None, gamma=1.0):
        super(EnergyLandscape, self).__init__()
        self.name = "EnergyLandscape"
        self.a = params
        self.gamma = gamma

    def forward(self, t, z):
        result = (
            torch.stack(
                [z[0][:, 1], -self.v_prime(z[0][:, 0]) - self.gamma * z[0][:, 1]]
            ).transpose(0, 1),
        )
        return result


class ThreeMinima(EnergyLandscape):
    def __init__(self, params, gamma=1.0):
        super(ThreeMinima, self).__init__(params, gamma)
        self.true_minima = [-0.850, 0.269, 1.839]
        self.true_a = [-0.192, 0.176, 2.364, -0.768, -2.5, 1.2]
        if params is None:
            self.a = self.true_a
        self.number_of_minima = len(self.true_minima)

    def v(self, x):
        return 0.2 * (x - 2) * (x - 0.5) * (x + 1) * (x + 0.6) * (x - 1.6) * x

    def v_prime(self, x):
        result = (
            self.a[5] * x ** 5
            + self.a[4] * x ** 4
            + self.a[3] * x ** 3
            + self.a[2] * x ** 2
            + self.a[1] * x
            + self.a[0]
        )
        return result


class ODEModel(torch.nn.Module):
    def __init__(
        self,
        ODE,
        params,
        step_size=1,
        tol=0.001,
        gamma=1.0,
        adjoint=False,
        solver="euler",
        grad_ode_params=True,
    ):
        super(ODEModel, self).__init__()
        if params is None:
            self.params = None
        else:
            self.params = torch.nn.Parameter(
                torch.tensor(params), requires_grad=grad_ode_params
            )
        self.step_size = step_size
        self.tol = tol
        if adjoint:
            self.ode_solver = odeint_adjoint
        else:
            self.ode_solver = odeint
        self.solver = solver
        self.func = ODE(params, gamma)

    def reset_params(self, params):
        self.params = torch.nn.Parameter(torch.tensor(params),)
        self.params.requires_grad = True
        print(self.params)

    def solve(self, z0, tf):
        z0 = (z0,)
        t = torch.arange(0, tf + self.step_size, self.step_size)
        out = self.ode_solver(
            self.func,
            z0,
            t,
            atol=self.tol,
            rtol=self.tol,
            method=self.solver,
            options={"step_size": self.step_size},
        )
        return out[0][-1]
