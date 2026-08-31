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
"""Thermodynamic models for the JAX solver.

JAX port of the TF thermodynamic models:
  - ConstantDensity: rho = const.
  - LinearMixing: rho = sum(Y_i * rho_i) based on mass fractions.
  - IdealGas: rho = p / (R * T), with geopotential reference state.

Each model provides:
  - rho_ref(ref_field): Reference density (constant or height-dependent).
  - p_ref(zz): Reference pressure (constant or height-dependent).
  - update_density(states, additional_states): Thermodynamic density.
"""

import abc
from typing import Optional

import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.physics.thermodynamics import utils as thermodynamics_utils
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

R_U = thermodynamics_utils.R_UNIVERSAL
G = thermodynamics_utils.G
INERT_SPECIES = thermodynamics_utils.INERT_SPECIES

# Variables that are not chemical species.
NON_SPECIES = ('T', 'theta')


class ThermodynamicModel(abc.ABC):
  """Abstract base class for thermodynamic models."""

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    self._params = params
    self._rho = params.rho

  def rho_ref(
      self,
      ref_field: ScalarField,
      additional_states: Optional[ScalarFieldMap] = None,
  ) -> ScalarField:
    """Returns the reference density field.

    Args:
      ref_field: A reference field for shape/device placement.
      additional_states: Helper variables (unused in base implementation).

    Returns:
      The reference density as a 3D array.
    """
    del additional_states
    return self._rho * jnp.ones_like(ref_field)

  def p_ref(
      self,
      zz: ScalarField,
      additional_states: Optional[ScalarFieldMap] = None,
  ) -> ScalarField:
    """Returns the reference pressure field.

    Args:
      zz: The geopotential height field.
      additional_states: Helper variables (unused in base implementation).

    Returns:
      The reference pressure as a 3D array.
    """
    del additional_states
    return self._params.p_thermal * jnp.ones_like(zz)

  @abc.abstractmethod
  def update_density(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the thermodynamic density from flow field variables.

    Args:
      states: Flow field variables.
      additional_states: Helper variables.

    Returns:
      The thermodynamic density field.
    """


class ConstantDensity(ThermodynamicModel):
  """Constant density model: rho = const."""

  def update_density(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Returns the constant density."""
    del additional_states
    return self._rho * jnp.ones_like(list(states.values())[0])


class LinearMixing(ThermodynamicModel):
  """Linear mixing rule: rho = sum(Y_i * rho_i).

  Each species has a constant density. The mixture density is the
  mass-fraction-weighted sum of species densities.
  """

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    super().__init__(params)
    self._rho_sc = {
        name: params.density(name)
        for name in params.scalars_names
        if name not in NON_SPECIES
    }
    self._rho_sc[INERT_SPECIES] = params.rho

  def update_density(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Updates density using the linear mixing rule."""
    del additional_states

    # Collect and regularize species mass fractions.
    scalars = {
        name: thermodynamics_utils.regularize_scalar_bound(states[name])
        for name in self._rho_sc
        if name != INERT_SPECIES and name in states
    }

    if scalars:
      scalars[INERT_SPECIES] = (
          thermodynamics_utils.compute_ambient_air_fraction(scalars)
      )
      sc_reg = thermodynamics_utils.regularize_scalar_sum(scalars)
    else:
      sc_reg = {INERT_SPECIES: jnp.ones_like(list(states.values())[0])}

    rho_mix = jnp.zeros_like(list(sc_reg.values())[0])
    for sc_name, sc_val in sc_reg.items():
      rho_mix = rho_mix + sc_val * self._rho_sc[sc_name]
    return rho_mix


class IdealGas(ThermodynamicModel):
  """Ideal gas model: rho = p / (R * T).

  Supports geopotential reference states with a parameterized virtual
  temperature profile, and potential temperature conversion.
  """

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    super().__init__(params)

    self._molecular_weights = {
        name: params.molecular_weight(name)
        for name in params.scalars_names
        if name not in NON_SPECIES
    }
    # Ensure the inert (ambient) species is always in the molecular weights
    # so the reciprocal mixing rule doesn't divide by zero.
    if INERT_SPECIES not in self._molecular_weights:
      self._molecular_weights[INERT_SPECIES] = (
          thermodynamics_utils.DRY_AIR_MOLECULAR_WEIGHT
      )
    self._p_thermal = params.p_thermal

    assert (
        model_params := params.thermodynamics
    ) is not None, 'Thermodynamics must be set in the config.'
    ig = model_params.ideal_gas_law
    self._t_s = ig.t_s
    w_inert = self._molecular_weights.get(
        INERT_SPECIES, thermodynamics_utils.DRY_AIR_MOLECULAR_WEIGHT
    )
    self.r_d = R_U / w_inert
    self.cp_d = ig.cv_d + self.r_d
    self.kappa = self.r_d / self.cp_d
    self._height = ig.height
    self._delta_t = ig.delta_t
    self._const_theta = ig.const_theta if ig.HasField('const_theta') else None

  @staticmethod
  def density_by_ideal_gas_law(
      p: ScalarField,
      r: float | ScalarField,
      t: ScalarField,
  ) -> ScalarField:
    """Computes density: rho = p / (r * T)."""
    return p / r / t

  def _potential_temperature_to_temperature(
      self,
      theta: ScalarField,
      zz: Optional[ScalarField] = None,
      additional_states: Optional[ScalarFieldMap] = None,
  ) -> ScalarField:
    """Converts potential temperature to temperature.

    T = theta * (p_ref / p_thermal)^kappa.

    Args:
      theta: Potential temperature (K).
      zz: Geopotential height (m).
      additional_states: Helper variables.

    Returns:
      Temperature (K).
    """
    if zz is None:
      return theta
    return (
        theta
        * (self.p_ref(zz, additional_states) / self._p_thermal) ** self.kappa
    )

  def temperature_to_potential_temperature(
      self,
      t: ScalarField,
      zz: Optional[ScalarField] = None,
      additional_states: Optional[ScalarFieldMap] = None,
  ) -> ScalarField:
    """Converts temperature to potential temperature.

    theta = T * (p_ref / p_thermal)^(-kappa).

    Args:
      t: Temperature (K).
      zz: Geopotential height (m).
      additional_states: Helper variables.

    Returns:
      Potential temperature (K).
    """
    if zz is None:
      return t
    return t * (self.p_ref(zz, additional_states) / self._p_thermal) ** (
        -self.kappa
    )

  def p_ref(
      self,
      zz: ScalarField,
      additional_states: Optional[ScalarFieldMap] = None,
  ) -> ScalarField:
    """Computes the reference pressure considering the geopotential.

    Assuming the virtual temperature profile takes the form:
    T = T_s - delta_T * tanh(z / H_t), the hydrostatic pressure is derived
    from the ideal gas law.

    For constant potential temperature, uses the isentropic relation:
    p(z) = p_s * (1 - g*z / (cp_d * theta_s))^(1/kappa).

    Args:
      zz: The geopotential height field.
      additional_states: Helper variables.

    Returns:
      The reference pressure as a function of height.
    """
    del additional_states

    if self._const_theta is not None:
      return self._p_thermal * (
          1.0 - G * zz / self.cp_d / self._const_theta
      ) ** (1.0 / self.kappa)

    delta_t_frac = self._delta_t / self._t_s
    h_sfc = self.r_d * self._t_s / G

    return self._p_thermal * jnp.exp(
        -(
            zz
            + self._height
            * delta_t_frac
            * (
                jnp.log(1.0 - delta_t_frac * jnp.tanh(zz / self._height))
                - jnp.log(1.0 + jnp.tanh(zz / self._height))
                + zz / self._height
            )
        )
        / h_sfc
        / (1.0 - delta_t_frac**2)
    )

  def t_ref(self, zz: Optional[ScalarField] = None) -> ScalarField:
    """Generates the reference temperature considering the geopotential.

    Args:
      zz: The geopotential height. If None, returns T_s as a scalar.

    Returns:
      The reference temperature as a function of height.
    """
    if zz is None:
      return jnp.float32(self._t_s)

    if self._const_theta is not None:
      theta = self._const_theta * jnp.ones_like(zz)
      return self._potential_temperature_to_temperature(theta, zz)

    return self._t_s - self._delta_t * jnp.tanh(zz / self._height)

  def rho_ref(
      self,
      ref_field: ScalarField,
      additional_states: Optional[ScalarFieldMap] = None,
  ) -> ScalarField:
    """Generates the reference density considering the geopotential.

    Args:
      ref_field: Reference field (used as zz if 'zz' is in additional_states,
        otherwise used for shape).
      additional_states: Helper variables. If 'zz' is present, it is used as the
        geopotential height.

    Returns:
      The reference density as a function of height.
    """
    if additional_states is not None and 'zz' in additional_states:
      zz = additional_states['zz']
    else:
      zz = jnp.zeros_like(ref_field)

    return self.density_by_ideal_gas_law(
        self.p_ref(zz, additional_states),
        self.r_d,
        self.t_ref(zz),
    )

  def update_density(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Updates the density with the ideal gas law.

    If temperature 'T' is in states, uses it directly.
    If potential temperature 'theta' is in states, converts to temperature.

    Args:
      states: Flow field variables. Must contain 'T' or 'theta'.
      additional_states: Helper variables. May contain 'zz'.

    Returns:
      The thermodynamic density field.

    Raises:
      ValueError: If neither 'T' nor 'theta' is found in states.
    """
    zz = additional_states.get('zz')
    if zz is None:
      zz = jnp.zeros_like(list(states.values())[0])

    if 'T' in states:
      t = states['T']
    elif 'theta' in states:
      t = self._potential_temperature_to_temperature(states['theta'], zz)
    else:
      raise ValueError(
          'Either temperature (T) or potential temperature (theta) is '
          'required for the ideal gas law.'
      )

    # Compute species mass fractions for mixture molecular weight.
    scalars = {
        name: thermodynamics_utils.regularize_scalar_bound(states[name])
        for name in self._molecular_weights
        if name != INERT_SPECIES and name in states
    }

    if scalars:
      scalars[INERT_SPECIES] = (
          thermodynamics_utils.compute_ambient_air_fraction(scalars)
      )
      sc_reg = thermodynamics_utils.regularize_scalar_sum(scalars)
    else:
      sc_reg = {INERT_SPECIES: jnp.ones_like(list(states.values())[0])}

    w_mix = thermodynamics_utils.compute_mixture_molecular_weight(
        self._molecular_weights, sc_reg
    )

    return self.density_by_ideal_gas_law(
        self.p_ref(zz, additional_states),
        R_U / w_mix,
        t,
    )
