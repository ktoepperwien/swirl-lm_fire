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

"""Potential temperature scalar model (JAX).

This module implements the `PotentialTemperature` scalar model for the
potential temperature equation. It extends `GenericScalarModel` with
specialised source terms for:

  - Large-scale subsidence (enabled via proto config)
  - Cloud radiation (requires Water thermodynamics model -- not yet ported)
  - Condensation/precipitation (requires microphysics -- not yet ported)

Supported scalar names: 'theta', 'theta_li'.
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

# Scalar names that map to the potential temperature model.
POTENTIAL_TEMPERATURE_VARNAMES = ('theta', 'theta_li')


class PotentialTemperature(scalar_model_generic.GenericScalarModel):
  """Scalar model for the potential temperature equation.

  Adds source terms for subsidence, radiation, and condensation
  on top of the generic scalar transport framework.
  """

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      scalar_name: str,
      sgs: sgs_model_lib.SgsModel | None = None,
  ):
    """Initialises the potential temperature model.

    Args:
      params: Simulation parameters.
      scalar_name: Name of the scalar ('theta' or 'theta_li').
      sgs: Optional SGS model for turbulent diffusivity.

    Raises:
      ValueError: If scalar_name is not a supported potential temperature type.
    """
    if scalar_name not in POTENTIAL_TEMPERATURE_VARNAMES:
      raise ValueError(
          f'Unsupported potential temperature scalar: {scalar_name}. '
          f'Supported types: {POTENTIAL_TEMPERATURE_VARNAMES}.'
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
        'potential_temperature'
    ):
      self._include_subsidence = (
          self._scalar_config.potential_temperature.include_subsidence
      )

    # Radiation source (requires Water model, not yet ported).
    self._include_radiation = False
    if self._scalar_config is not None and self._scalar_config.HasField(
        'potential_temperature'
    ):
      self._include_radiation = (
          self._scalar_config.potential_temperature.include_radiation
      )
      if self._include_radiation:
        logging.warning(
            'Radiation source for potential temperature is not yet '
            'implemented in the JAX solver. It will be ignored.'
        )

    if self._include_subsidence and self._g_dim is None:
      raise ValueError(
          'Gravity dimension (g_dim) must be defined to include subsidence '
          'in the potential temperature equation.'
      )

  def source_fn(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes source terms for the potential temperature equation.

    Currently supported sources:
      - Subsidence: -rho * w_s * d(theta)/dz, where w_s is the
        large-scale subsidence velocity.

    Args:
      phi: Current potential temperature field.
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
