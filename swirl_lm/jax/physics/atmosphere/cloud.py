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
"""A class for handling cloud specific parametrizations (JAX port).

This is the JAX port of `swirl_lm.physics.atmosphere.cloud`.
"""


from collections.abc import Sequence

import jax.numpy as jnp
from swirl_lm.jax.physics.thermodynamics import water
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField

# Parameters required by the radiation model. Reference:
# Stevens, Bjorn, et al. 2005. "Evaluation of Large-Eddy Simulations via
# Observations of Nocturnal Marine Stratocumulus." Monthly Weather Review
# 133 (6): 1443-62.
_F0 = 70.0
_F1 = 22.0
_KAPPA = 85.0
_ALPHA_Z = 1.0
# The subsidence velocity coefficient.
_D = 3.75e-6
# The initial height of the cloud, in units of m.
_ZI = 840.0


class Cloud:
  """An object for handling cloud parametrizations."""

  def __init__(self, water_model: water.Water) -> None:
    """Initialize with a water thermodynamics model."""
    self._water_model = water_model

  @property
  def water_model(self) -> water.Water:
    """The underlying water thermodynamics model."""
    return self._water_model

  def _radiation(
      self,
      q_h: ScalarField,
      q_l: ScalarField,
      rho: ScalarField,
      z: ScalarField,
  ) -> ScalarField:
    """Computes the radiation term based on given parameters.

    Args:
      q_h: The integral of the liquid water specific mass from `z` to the
        maximum height of the simulation.
      q_l: The integral of the liquid water specific mass from 0 to `z`.
      rho: The density of air at `z`.
      z: The current height.

    Returns:
      The radiation source term.
    """
    return (
        _F0 * jnp.exp(-_KAPPA * q_h)
        + _F1 * jnp.exp(-_KAPPA * q_l)
        + rho
        * self._water_model.cp_d
        * _D
        * _ALPHA_Z
        * (
            0.25 * jnp.power(jnp.maximum(z - _ZI, 0.0), 4.0 / 3.0)
            + _ZI * jnp.power(jnp.maximum(z - _ZI, 0.0), 1.0 / 3.0)
        )
    )

  def source_by_radiation(
      self,
      q_l: ScalarField,
      rho: ScalarField,
      zz: ScalarField,
      h: float,
      g_dim: int,
      halos: Sequence[int],
  ) -> ScalarField:
    """Computes the energy source term due to radiation.

    Reference:
    Stevens, Bjorn, et al. 2005. "Evaluation of Large-Eddy Simulations via
    Observations of Nocturnal Marine Stratocumulus." Monthly Weather Review
    133 (6): 1443-62.

    In single-device mode, vertical integration is done locally.

    Args:
      q_l: The liquid humidity.
      rho: The density.
      zz: The height values.
      h: The vertical grid spacing.
      g_dim: The dimension of the gravity.
      halos: A sequence of int representing the halo points in each dimension.

    Returns:
      The source term in the total energy equation due to radiation.
    """
    # Zero out halos in the vertical direction for clean integration.
    hw = halos[g_dim]
    rho_q_l = rho * q_l
    # Zero out vertical halos.
    if hw > 0:
      slices_before = [slice(None)] * 3
      slices_after = [slice(None)] * 3
      slices_before[g_dim] = slice(None, hw)
      slices_after[g_dim] = slice(-hw, None)
      rho_q_l = rho_q_l.at[tuple(slices_before)].set(0.0)
      rho_q_l = rho_q_l.at[tuple(slices_after)].set(0.0)

    # Compute cumulative integrals along g_dim.
    # q_below: integral from 0 to z.
    q_below = jnp.cumsum(rho_q_l * h, axis=g_dim)
    # q_above: integral from z to top (total - q_below).
    total = jnp.sum(rho_q_l * h, axis=g_dim, keepdims=True)
    q_above = total - q_below

    return self._radiation(q_above, q_below, rho, zz)
