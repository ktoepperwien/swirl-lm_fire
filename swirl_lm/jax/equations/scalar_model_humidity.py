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

"""Humidity scalar model (JAX).

This module implements the `Humidity` scalar model for humidity transport
equations. It extends `GenericScalarModel` with specialised source terms for:

  - Large-scale subsidence (enabled via proto config)
  - Precipitation (requires Water model + microphysics -- not yet ported)
  - Condensation (requires Water model + microphysics -- not yet ported)
  - Sedimentation (requires microphysics -- not yet ported)

Supported scalar names: 'q_t', 'q_r', 'q_s', 'q_c', 'q_v'.
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

# Scalar names that map to the humidity model.
HUMIDITY_VARNAMES = ('q_t', 'q_r', 'q_s', 'q_c', 'q_v')


class Humidity(scalar_model_generic.GenericScalarModel):
  """Scalar model for humidity transport equations.

  Adds source terms for subsidence, precipitation, condensation,
  and sedimentation on top of the generic scalar transport framework.
  """

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      scalar_name: str,
      sgs: sgs_model_lib.SgsModel | None = None,
  ):
    """Initialises the humidity model.

    Args:
      params: Simulation parameters.
      scalar_name: Name of the scalar (one of HUMIDITY_VARNAMES).
      sgs: Optional SGS model for turbulent diffusivity.

    Raises:
      ValueError: If scalar_name is not a supported humidity type.
    """
    if scalar_name not in HUMIDITY_VARNAMES:
      raise ValueError(
          f'Unsupported humidity scalar: {scalar_name}. '
          f'Supported types: {HUMIDITY_VARNAMES}.'
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

    # Subsidence source (only for q_t).
    self._include_subsidence = False
    if self._scalar_config is not None and self._scalar_config.HasField(
        'humidity'
    ):
      self._include_subsidence = (
          self._scalar_config.humidity.include_subsidence
          and self._g_dim is not None
      )

    # Log warnings for unported features.
    if params.microphysics is not None:
      if params.microphysics.include_precipitation:
        logging.warning(
            'Precipitation source for humidity is not yet implemented in '
            'the JAX solver. It will be ignored.'
        )
      if params.microphysics.include_condensation:
        logging.warning(
            'Condensation source for humidity is not yet implemented in '
            'the JAX solver. It will be ignored.'
        )

  def source_fn(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes source terms for the humidity equation.

    Currently supported sources:
      - Subsidence for q_t: -rho * w_s * d(q_c)/dz.

    Args:
      phi: Current humidity field.
      states: Flow field variables (must include 'rho').
      additional_states: Helper variables (must include 'zz' for subsidence).

    Returns:
      The total source term field.
    """
    source = jnp.zeros_like(phi)

    if self._include_subsidence and self._scalar_name == 'q_t':
      zz = additional_states.get('zz', jnp.zeros_like(phi))
      # For q_t subsidence, the TF code uses q_c (condensed water) as the
      # field for the vertical derivative, not q_t itself.
      q_c = additional_states.get('q_c', jnp.zeros_like(phi))
      axes = self._params.grid_params.data_axis_order
      assert self._g_dim is not None
      g_axis = axes[self._g_dim]
      src_subsidence = eq_utils.source_by_subsidence_velocity(
          self._deriv_lib,
          states[common.KEY_RHO],
          zz,
          q_c,
          g_axis,
          additional_states,
      )
      source = source + src_subsidence

    return source
