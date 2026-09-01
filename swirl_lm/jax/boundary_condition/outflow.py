# Copyright 2026 The swirl_lm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library for the outflow boundary condition (JAX port).

This is the JAX port of `swirl_lm.boundary_condition.outflow`. It applies a
forward Euler + upwinding scheme to solve the outflow boundary equation:
  dφ/dt = -max(u) dφ/dx.
The outflow velocity is rescaled so that mass flux at the outlet matches the
inlet.
"""


from collections.abc import Callable
import re

import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# JAX equivalent of StatesUpdateFn: no kernel_op, replica_id, replicas.
StatesUpdateFn = Callable[
    [ScalarFieldMap, ScalarFieldMap, parameters_lib.SwirlLMParameters],
    dict[str, ScalarField],
]


def outflow_boundary_condition() -> StatesUpdateFn:
  r"""Generates an update function for an outflow boundary condition.

  A forward Euler with upwinding scheme is used to solve the outflow boundary
  equation:
    ∂ϕ/∂t = -max(u) ∂ϕ/∂x.
  In discrete form:
    ϕⱼⁿ⁺¹ = (1 - Δt max(u)/Δx) ϕⱼⁿ + Δt max(u)/Δx) ϕⱼ₋₁ⁿ.
  The outflow velocity is rescaled so that the mass flux at the outlet is the
  same as the inlet. Note that this boundary condition can only be applied in
  the variable density solver where `rho` is in `states`.

  Returns:
    A function that updates the Dirichlet boundary condition for required
    variables in dimension 0 on face 1, i.e. all `additional_states` with key
    regular expression 'bc_(\w+)_0_1', with `\w+` being the variable name.
  """

  def get_boundary_update_fn(
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      params: parameters_lib.SwirlLMParameters,
  ) -> dict[str, ScalarField]:
    """Computes the boundary condition for variables on the +x boundary."""
    gp = params.grid_params
    hw = gp.halo_width

    u_max = jnp.max(states['u'][:, -hw - 1, :])
    dx = gp.dx
    assert (
        dx is not None
    ), 'Outflow BC requires uniform grid (dx must not be None).'
    cfl = gp.dt * u_max / dx
    coeff = 1.0 - cfl

    def mass_flux_x_face(face_index: int) -> ScalarField:
      """Computes the mass flux in x face at `face_index`."""
      return jnp.sum(
          states['rho'][:, face_index, :] * states['u'][:, face_index, :]
      )

    mass_exit = mass_flux_x_face(-hw - 1)
    mass_inlet = mass_flux_x_face(hw)
    mass_correction = mass_inlet / mass_exit

    def update_boundary_values(var_name: str) -> ScalarField:
      """Update the boundary values for variable `var_name`."""
      bc_name = f'bc_{var_name}_0_1'
      correction_factor = mass_correction if var_name == 'u' else 1.0
      return correction_factor * (
          coeff * additional_states[bc_name]
          + (1.0 - coeff) * states[var_name][:, -hw - 1 : -hw, :]
      )

    return {
        key: (
            update_boundary_values(re.split(r'bc_(\w+)_0_1', key)[1])
            if re.search(r'bc_(\w+)_0_1', key)
            else val
        )
        for key, val in additional_states.items()
    }

  return get_boundary_update_fn
