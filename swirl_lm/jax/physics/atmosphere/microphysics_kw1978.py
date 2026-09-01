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
"""A library of the microphysics from Klemp & Wilhelmson, 1978 (JAX port).

JAX port of `swirl_lm.physics.atmosphere.microphysics_kw1978`.
"""


from typing import Optional

import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.physics.atmosphere import microphysics_generic
from swirl_lm.jax.physics.thermodynamics import water
from swirl_lm.jax.utility import types
from swirl_lm.physics import constants

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# The key used for density in states.
_KEY_RHO = 'rho'


class MicrophysicsKW1978(microphysics_generic.Microphysics):
  """An object for handling precipitation modeling."""

  def evaporation(
      self,
      rho: ScalarField,
      temperature: ScalarField,
      q_r: ScalarField,
      q_v: ScalarField,
      q_l: ScalarField,
      q_c: ScalarField,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    r"""The rain evaporation rate.

    Based on Klemp & Wilhelmson, 1978, eqs. (2.14a) & (2.14b).

    Args:
      rho: The moist air density, in kg/m^3.
      temperature: The temperature.
      q_r: The rain water mixture fraction (kg/kg).
      q_v: The cloud vapor fraction (kg/kg).
      q_l: The specific humidity of the cloud liquid phase (kg/kg).
      q_c: The specific humidity of the cloud humidity condensed phase (kg/kg).
      additional_states: Helper variables in the simulation.

    Returns:
      The evaporation rate of the rain drops in 1/sec.
    """
    zz = additional_states.get('zz', jnp.zeros_like(q_r))
    rho_bar = self.water_model.rho_ref(zz, additional_states)
    p_bar = self.water_model.p_ref(zz, additional_states)
    q_vs = self.water_model.saturation_q_vapor(temperature, rho, q_l, q_c)
    p_vs = q_vs * p_bar

    precipitation_bulk_density = rho_bar * jnp.clip(q_r, 0.0, 1.0)
    c = 1.6 + 30.3922 * jnp.power(precipitation_bulk_density, 0.2046)

    e_r = (
        (1.0 / rho_bar)
        * (1.0 - q_v / q_vs)
        * c
        * jnp.power(precipitation_bulk_density, 0.525)
        / (2.03e4 + 9.584e6 / p_vs)
    )
    return e_r

  def autoconversion_and_accretion(
      self,
      q_r: ScalarField,
      q_l: ScalarField,
  ) -> ScalarField:
    r"""The conversion rate from cloud liquid humidity to rain water.

    Based on Klemp & Wilhelmson, 1978, eqs. (2.13a) & (2.13b).

    Args:
      q_r: The rain water mixture fraction (kg/kg).
      q_l: The specific humidity of the cloud liquid phase (kg/kg).

    Returns:
      The conversion rate of cloud liquid phase humidity to rain drops (1/sec).
    """
    a = 0.001
    k_1 = 0.001
    k_2 = 2.2

    a_r = k_1 * jnp.maximum(q_l - a, 0.0)
    c_r = k_2 * q_l * jnp.power(jnp.clip(q_r, 0.0, 1.0), 0.875)
    return a_r + c_r

  def terminal_velocity(
      self,
      rho: ScalarField,
      q_r: ScalarField,
      additional_states: ScalarFieldMap,
      rho_ref: float = 1.15,
  ) -> ScalarField:
    r"""Terminal velocity used for rain water convection term.

    Based on Klemp & Wilhelmson, 1978, eq. (2.15).

    Args:
       rho: density [kg/m^3] (unused, kept for API compatibility).
       q_r: rain water mixture fraction [kg/kg].
       additional_states: Helper variables in the simulation.
       rho_ref: Reference density at ground level. Default is 1.15 [kg/m^3].

    Returns:
      The terminal velocity of rain water in m/s.
    """
    del rho
    zz = additional_states.get('zz', jnp.zeros_like(q_r))
    rho_bar = self.water_model.rho_ref(zz, additional_states)

    k1 = 14.34
    k2 = 0.1346
    sqrt_rho_ref = jnp.sqrt(rho_ref)

    f1 = k1 * jnp.power(rho_bar * jnp.clip(q_r, 0.0, 1.0), k2)
    f2 = sqrt_rho_ref / jnp.sqrt(rho_bar)
    return f1 * f2


class Adapter(microphysics_generic.MicrophysicsAdapter):
  """Interface between the Kessler microphysics model and rest of Swirl-LM."""

  _kessler: MicrophysicsKW1978

  def __init__(
      self,
      params: parameters_lib.SwirlLMParameters,
      water_model: water.Water,
  ):
    assert params.microphysics is not None and params.microphysics.HasField(
        'kessler'
    ), 'The microphysics.kessler field needs to be set in SwirlLMParameters.'
    self._kessler_params = params.microphysics.kessler
    self._kessler = MicrophysicsKW1978(params, water_model)

  def terminal_velocity(
      self,
      varname: str,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the terminal velocity for `q_r`, `q_s`, `q_l`, or `q_i`."""
    assert varname in ('q_r', 'q_s', 'q_l', 'q_i'), (
        f'Terminal velocity is for q_r/q_s/q_l/q_i only, but {varname}'
        ' is provided.'
    )
    return self._kessler.terminal_velocity(
        states['rho_thermal'], states[varname], additional_states
    )

  def humidity_source_fn(
      self,
      varname: str,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      thermo_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the source term in a humidity equation."""
    q_r = thermo_states['q_r']
    q_l = thermo_states['q_l']
    aut_and_acc = self._kessler.autoconversion_and_accretion(q_r, q_l)
    rain_water_evaporation_rate = self._kessler.evaporation(
        states['rho_thermal'],
        thermo_states['T'],
        q_r,
        thermo_states['q_v'],
        q_l,
        thermo_states['q_c'],
        additional_states,
    )
    net_cloud_liquid_to_rain_water_rate = (
        aut_and_acc - rain_water_evaporation_rate
    )
    cloud_liquid_to_water_source = (
        net_cloud_liquid_to_rain_water_rate * states[_KEY_RHO]
    )
    if varname == 'q_t':
      return -cloud_liquid_to_water_source
    elif varname == 'q_r':
      return cloud_liquid_to_water_source
    elif varname == 'q_c':
      return -(aut_and_acc * states[_KEY_RHO])
    elif varname == 'q_v':
      return rain_water_evaporation_rate * states[_KEY_RHO]
    else:
      raise NotImplementedError(
          f'Precipitation for {varname} is not implemented. Only'
          ' q_t, q_r, q_c and q_v are supported.'
      )

  def potential_temperature_source_fn(
      self,
      varname: str,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      thermo_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the source term in a potential temperature equation."""
    assert varname in ('theta', 'theta_li'), (
        'Only theta and theta_li are supported for the microphysics source'
        f' term computation, but {varname} is provided.'
    )

    source = -self._kessler.evaporation(
        states['rho_thermal'],
        thermo_states['T'],
        states['q_r'],
        thermo_states['q_v'],
        thermo_states['q_l'],
        thermo_states['q_c'],
        additional_states,
    )

    if varname == 'theta_li':
      source = source + self._kessler.autoconversion_and_accretion(
          states['q_r'], thermo_states['q_l']
      )

    t_0 = self._kessler.water_model.t_ref(
        thermo_states['zz'], additional_states
    )
    zeros = jnp.zeros_like(t_0)
    theta_0 = self._kessler.water_model.temperature_to_potential_temperature(
        'theta',
        t_0,
        zeros,
        zeros,
        zeros,
        thermo_states['zz'],
        additional_states,
    )
    cp = self._kessler.water_model.cp_m(
        thermo_states['q_t'], thermo_states['q_l'], thermo_states['q_i']
    )

    return (
        states['rho']
        * (self._kessler.water_model.lh_v0 / cp)
        * (theta_0 / t_0)
        * source
    )

  def total_energy_source_fn(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      thermo_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the source term in the total energy equation."""
    cloud_liquid_to_rain_water_rate = (
        self._kessler.autoconversion_and_accretion(
            thermo_states['q_r'], thermo_states['q_l']
        )
    )
    rain_water_evaporation_rate = self._kessler.evaporation(
        states['rho_thermal'],
        thermo_states['T'],
        thermo_states['q_r'],
        thermo_states['q_v'],
        thermo_states['q_l'],
        thermo_states['q_c'],
        additional_states,
    )
    pe = constants.G * thermo_states['zz']
    source_v = (
        (thermo_states['e_v'] + pe)
        * states['rho']
        * (-rain_water_evaporation_rate)
    )
    source_l = (
        (thermo_states['e_l'] + pe)
        * states['rho']
        * cloud_liquid_to_rain_water_rate
    )
    return source_v + source_l

  def condensation(
      self,
      rho: ScalarField,
      temperature: ScalarField,
      q_v: ScalarField,
      q_l: ScalarField,
      q_c: ScalarField,
      zz: Optional[ScalarField] = None,
      additional_states: Optional[ScalarFieldMap] = None,
  ) -> ScalarField:
    """Computes the condensation rate using Bryan & Fritsch (2002)."""
    return self._kessler.condensation_bf2002(
        rho,
        temperature,
        q_v,
        q_l,
        q_c,
        zz,
        additional_states,
    )
