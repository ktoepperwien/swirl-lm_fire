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

"""Total energy scalar model (JAX).

This module implements the `TotalEnergy` scalar model for the total
energy transport equation (`e_t`). It extends `GenericScalarModel` with
specialised source terms for:

  - Large-scale subsidence (enabled via proto config)
  - Viscous heating / pressure work (requires full thermodynamics)
  - Cloud radiation (requires Water model -- not yet ported)
  - Precipitation (requires microphysics -- not yet ported)

The full TotalEnergy model in TF requires the Water thermodynamics model.
This JAX version provides subsidence support and stubs the remaining sources.
"""

from __future__ import annotations

from absl import logging
import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.equations import common
from swirl_lm.jax.equations import scalar_model_generic
from swirl_lm.jax.equations import utils as eq_utils
from swirl_lm.jax.physics.turbulence import sgs_model as sgs_model_lib
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

TOTAL_ENERGY_VARNAME = 'e_t'


class TotalEnergy(scalar_model_generic.GenericScalarModel):
  """Scalar model for the total energy equation.

  Adds source terms for subsidence, viscous heating, radiation,
  and precipitation on top of the generic scalar transport framework.
  """

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      scalar_name: str,
      sgs: sgs_model_lib.SgsModel | None = None,
  ):
    """Initialises the total energy model.

    Args:
      params: Simulation parameters.
      scalar_name: Name of the scalar (must be 'e_t').
      sgs: Optional SGS model for turbulent diffusivity.

    Raises:
      ValueError: If scalar_name is not 'e_t'.
    """
    if scalar_name != TOTAL_ENERGY_VARNAME:
      raise ValueError(
          f'TotalEnergy is for {TOTAL_ENERGY_VARNAME} only, '
          f'but {scalar_name} is provided.'
      )
    super().__init__(params, scalar_name, sgs)

    self._deriv_lib = params.deriv_lib
    self._g_dim = params.g_dim

    # Parse proto config for this scalar.
    self._scalar_config = None
    for sc in params.scalars:
      if sc.name == scalar_name:
        self._scalar_config = sc
        break

    # Subsidence source.
    self._include_subsidence = False
    if self._scalar_config is not None and self._scalar_config.HasField(
        'total_energy'
    ):
      self._include_subsidence = (
          self._scalar_config.total_energy.include_subsidence
          and self._g_dim is not None
      )

    # Log warnings for unported features.
    if (
        self._scalar_config is not None
        and self._scalar_config.HasField('total_energy')
        and self._scalar_config.total_energy.include_radiation
    ):
      logging.warning(
          'Radiation source for total energy is not yet implemented in '
          'the JAX solver. It will be ignored.'
      )

  def source_fn(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes source terms for the total energy equation.

    Currently supported sources:
      - Subsidence: -rho * w_s * d(e_t)/dz.

    Args:
      phi: Current total energy field.
      states: Flow field variables (must include 'rho').
      additional_states: Helper variables (must include 'zz' for subsidence).

    Returns:
      The total source term field.
    """
    source = jnp.zeros_like(phi)

    if self._include_subsidence:
      zz = additional_states.get('zz', jnp.zeros_like(phi))
      axes = self._params.grid_params.data_axis_order
      assert self._g_dim is not None
      g_axis = axes[self._g_dim]
      src_subsidence = eq_utils.source_by_subsidence_velocity(
          self._deriv_lib,
          states[common.KEY_RHO],
          zz,
          phi,
          g_axis,
          additional_states,
      )
      source = source + src_subsidence

    return source
