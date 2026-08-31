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

"""Scalar model interface and generic implementation (JAX).

This module defines the `ScalarModel` protocol -- the interface all scalar
transport models must satisfy -- and a `GenericScalarModel` implementation
for passive scalars with no specialised source terms.

The key abstraction:

  Every scalar needs three ingredients for its RHS:
    1. Diffusivity -- `get_diffusivity()`
    2. Source term  -- `source_fn()`
    3. (Optionally) custom convection/diffusion scalars

Specialised models (PotentialTemperature, Humidity, TotalEnergy) subclass
`GenericScalarModel` and override the methods that differ from the generic
behaviour, typically `source_fn()`.
"""

from __future__ import annotations

import abc

import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.equations import common
from swirl_lm.jax.physics.turbulence import sgs_model as sgs_model_lib
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


class ScalarModel(abc.ABC):
  """Interface for scalar transport models.

  Each scalar variable in the simulation has an associated `ScalarModel` that
  provides its physical diffusivity, source terms, and (optionally) custom
  scalars for convection and diffusion.
  """

  @abc.abstractmethod
  def get_diffusivity(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Returns the effective diffusivity field for this scalar.

    This includes both molecular diffusivity and, if SGS is active, the
    turbulent contribution.

    Args:
      phi: Current scalar field value.
      states: All flow field variables.
      additional_states: Helper variables.

    Returns:
      The effective diffusivity field.
    """

  @abc.abstractmethod
  def source_fn(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the source term for this scalar transport equation.

    Args:
      phi: Current scalar field value.
      states: All flow field variables.
      additional_states: Helper variables.

    Returns:
      The source term field. A zero field for passive scalars.
    """

  def mass_source_fn(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField | None:
    """Computes the mass source for the pressure solver (Low Mach only).

    Some scalar models (e.g., humidity with evaporation/condensation) produce
    a mass source that must be added to the pressure Poisson equation RHS.

    Args:
      phi: Current scalar field value.
      states: All flow field variables.
      additional_states: Helper variables.

    Returns:
      The mass source field, or None if this scalar does not contribute.
    """
    del phi, states, additional_states
    return None

  def get_scalar_for_convection(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Returns the scalar value to use in the convection term.

    Override for scalars that need a modified convection variable (e.g.,
    humidity in conservative form may use q_t instead of q_v).

    Args:
      phi: Current scalar field value.
      states: All flow field variables.
      additional_states: Helper variables.

    Returns:
      The scalar value for convection (default: phi itself).
    """
    del states, additional_states
    return phi

  def get_scalar_for_diffusion(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Returns the scalar value to use in the diffusion term.

    Override for scalars where the diffused variable differs from the
    transported one.

    Args:
      phi: Current scalar field value.
      states: All flow field variables.
      additional_states: Helper variables.

    Returns:
      The scalar value for diffusion (default: phi itself).
    """
    del states, additional_states
    return phi

  def get_momentum_for_convection(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> tuple[ScalarField, ScalarField, ScalarField]:
    """Returns the momentum components for the convection term.

    Override for scalars that use a different momentum definition
    (e.g., humidity uses rho*u instead of just the velocity for
    conservative advection).

    Args:
      phi: Current scalar field value.
      states: All flow field variables.
      additional_states: Helper variables.

    Returns:
      Tuple of (rho_u, rho_v, rho_w) momentum components.
    """
    del phi, additional_states
    return (
        states[common.KEY_RHO_U],
        states[common.KEY_RHO_V],
        states[common.KEY_RHO_W],
    )


class GenericScalarModel(ScalarModel):
  """Default scalar model for passive scalars (no specialised source terms).

  Provides molecular + SGS turbulent diffusivity and a zero source term.
  """

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      scalar_name: str,
      sgs: sgs_model_lib.SgsModel | None = None,
  ):
    """Initialises the generic scalar model.

    Args:
      params: Simulation parameters.
      scalar_name: Name of the scalar variable (must match proto config).
      sgs: Optional SGS model for turbulent diffusivity.
    """
    self._params = params
    self._scalar_name = scalar_name
    self._sgs = sgs
    self._molecular_diffusivity = params.diffusivity(scalar_name)

  @property
  def scalar_name(self) -> str:
    """Name of the scalar this model is associated with."""
    return self._scalar_name

  def get_diffusivity(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes molecular + SGS turbulent diffusivity."""
    diff = self._molecular_diffusivity * jnp.ones_like(phi)

    if self._sgs is not None:
      velocity = (
          states[common.KEY_U],
          states[common.KEY_V],
          states[common.KEY_W],
      )
      diff_t = self._sgs.turbulent_diffusivity(
          (phi,), additional_states, velocity
      )
      diff = diff + diff_t

    return diff

  def source_fn(
      self,
      phi: ScalarField,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Returns zero source for passive scalars."""
    del states, additional_states
    return jnp.zeros_like(phi)
