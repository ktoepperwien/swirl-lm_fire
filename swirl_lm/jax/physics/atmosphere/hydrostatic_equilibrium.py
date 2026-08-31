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

# Copyright 2022 Google LLC
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
"""Defines a utility class for computing hydrostatic states (JAX port).

This is the JAX port of
`swirl_lm.physics.atmosphere.hydrostatic_equilibrium`.

The hydrostatic pressure is computed via numerical integration of a function
of temperature, so its accuracy depends on the grid resolution and on the
smoothness of the given temperature profile.
"""


import enum

import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.utility import types
from swirl_lm.physics import constants

ScalarField = types.ScalarField


class InputType(enum.Enum):
  """Defines the name of a temperature variable."""

  # The temperature of dry air, or virtual temperature of moist air.
  TEMPERATURE = 'temperature'
  # The potential temperature of dry air, or virtual potential temperature of
  # moist air.
  POTENTIAL_TEMPERATURE = 'potential_temperature'


class HydrostaticEquilibrium:
  """Utility functions for computing states in hydrostatic equilibrium.

  In many applications, the background state or initial state of a geophysical
  flow simulation is specified via a realistic atmospheric temperature profile
  from which it is possible to derive a corresponding hydrostatically balanced
  pressure and density.

  Using this library, the hydrostatic pressure can be computed either from a
  (virtual) temperature or from a (virtual) potential temperature profile.
  """

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    """Initializes the hydrostatic equilibrium utility.

    Args:
      params: The simulation parameters.
    """
    self._params = params
    gp = params.grid_params
    g_vec = params.gravity_direction
    # Find the direction of gravity. Only vector along a particular dimension
    # is supported currently.
    self._g_dim: int | None = None
    for i in range(3):
      if np.abs(np.abs(g_vec[i]) - 1.0) < np.finfo(np.float32).resolution:
        self._g_dim = i
        break
    assert (
        self._g_dim is not None
    ), 'No gravity-aligned dimension found in the gravity direction.'
    dh_vals = (gp.dx, gp.dy, gp.dz)
    dh = dh_vals[self._g_dim]
    assert (
        dh is not None
    ), f'Grid spacing along gravity dim {self._g_dim} must not be None.'
    self._dh: float = dh
    self._halo_width = gp.halo_width

  def _temperature_integration_fn(self, t: ScalarField) -> ScalarField:
    """Computes the integrand for pressure from T: G / R / T(z)."""
    return constants.G / constants.R_D / t

  def _theta_integration_fn(self, t: ScalarField) -> ScalarField:
    """Computes the integrand for pressure from theta: G / Cp / theta(z)."""
    return constants.G / constants.CP / t

  def _p_fn_from_temperature(self, integral: ScalarField) -> ScalarField:
    """Computes pressure given an integral in terms of temperature."""
    return self._params.p_thermal * jnp.exp(-integral)

  def _p_fn_from_theta(self, integral: ScalarField) -> ScalarField:
    """Computes pressure given an integral in terms of theta."""
    return self._params.p_thermal * (1.0 - integral) ** (
        constants.CP / constants.R_D
    )

  def pressure(
      self,
      varname: str,
      t: ScalarField,
  ) -> ScalarField:
    """Computes the hydrostatic pressure from a given profile of T or theta.

    Note that the pressure calculation assumes the air is dry. If accounting
    for moisture is desired, the caller should pass the virtual temperature
    or virtual potential temperature instead.

    Args:
      varname: One of 'temperature' or 'potential_temperature'.
      t: A temperature, or potential temperature, field in units of K.

    Returns:
      The hydrostatic pressure that conforms to the given (potential)
      temperature profile.

    Raises:
      ValueError: If `varname` is not recognized.
    """
    if varname == InputType.POTENTIAL_TEMPERATURE.value:
      integration_fn = self._theta_integration_fn
      p_fn = self._p_fn_from_theta
    elif varname == InputType.TEMPERATURE.value:
      integration_fn = self._temperature_integration_fn
      p_fn = self._p_fn_from_temperature
    else:
      raise ValueError(
          f'{varname} is not a valid variable for hydrostatic pressure '
          f'computation. Available options are: '
          f"'{InputType.TEMPERATURE.value}' "
          f"or '{InputType.POTENTIAL_TEMPERATURE.value}'."
      )

    assert self._g_dim is not None

    # Strip halos in the vertical direction.
    hw = self._halo_width
    slices = [slice(None)] * 3
    slices[self._g_dim] = slice(hw, -hw) if hw > 0 else slice(None)
    t_interior = t[tuple(slices)]

    # Compute integrand and cumulative integral along g_dim.
    integrand = integration_fn(t_interior)
    integral = jnp.cumsum(integrand * self._dh, axis=self._g_dim)

    # Compute pressure in the interior.
    p_interior = p_fn(integral)

    # Pad back to original shape with zeros.
    pad_widths = [(0, 0)] * 3
    pad_widths[self._g_dim] = (hw, hw)
    p = jnp.pad(p_interior, pad_widths, mode='edge')

    return p
