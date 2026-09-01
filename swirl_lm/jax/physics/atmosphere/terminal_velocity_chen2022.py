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

# Copyright 2023 Google LLC
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
"""Data class wrapping terminal velocity parameterization from Chen et al.

JAX port of `swirl_lm.physics.atmosphere.terminal_velocity_chen2022`.
"""


import dataclasses

import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.physics.atmosphere import particles
from swirl_lm.jax.utility import types
from swirl_lm.physics.atmosphere import microphysics_one_moment_constants as constants
from swirl_lm.physics.atmosphere import microphysics_pb2
from swirl_lm.physics.atmosphere import terminal_velocity_chen2022_tables

Rain = particles.Rain
Snow = particles.Snow
Ice = particles.Ice
ScalarField = types.ScalarField

TableB1Coeffs = terminal_velocity_chen2022_tables.TableB1Coeffs
TableB3Coeffs = terminal_velocity_chen2022_tables.TableB3Coeffs
TableB5Coeffs = terminal_velocity_chen2022_tables.TableB5Coeffs


@dataclasses.dataclass(frozen=True)
class IceVelocity:
  """Computed parameters from table B3 of Chen et al. (2022)."""

  a: jnp.ndarray
  b: jnp.ndarray
  c: jnp.ndarray
  e: jnp.ndarray
  f: jnp.ndarray
  g: jnp.ndarray


@dataclasses.dataclass(frozen=True)
class SnowVelocity:
  """Computed parameters from table B5 of Chen et al. (2022)."""

  a: jnp.ndarray
  b: jnp.ndarray
  c: jnp.ndarray
  e: jnp.ndarray
  f: jnp.ndarray
  g: jnp.ndarray
  h: jnp.ndarray


@dataclasses.dataclass(frozen=True)
class TerminalVelocityCoefficients:
  """Final coefficients to be used in the terminal velocity equation."""

  a: tuple[jnp.ndarray, ...]
  b: tuple[jnp.ndarray, ...]
  c: tuple[jnp.ndarray, ...]


@dataclasses.dataclass(frozen=True)
class TerminalVelocityChen2022:
  """Coefficients for gamma-type terminal velocity parameterization."""

  _CLOUD_DROPLET_CORRECTION_FACTOR = 0.1
  _KAPPA = 1 / 3

  _rain: Rain
  _snow: Snow
  _ice: Ice

  rain_velocity: TableB1Coeffs
  ice_velocity: IceVelocity
  snow_velocity: SnowVelocity

  @classmethod
  def _precompute_ice_coeffs(cls, ice: Ice) -> IceVelocity:
    """Precomputes the coefficient formulas from table B3 of Chen et al."""
    rho = ice.params.rho
    table_b3 = TableB3Coeffs()
    a = (
        table_b3.a[0]
        + table_b3.a[1] * np.log(rho) ** 2
        + table_b3.a[2] * np.log(rho)
    )
    b = (
        table_b3.b[0]
        + table_b3.b[1] * np.log(rho)
        + table_b3.b[2] / np.sqrt(rho)
    ) ** -1
    c = (
        table_b3.c[0]
        + table_b3.c[1] * np.exp(table_b3.c[2] * rho)
        + table_b3.c[3] * np.sqrt(rho)
    )
    e = (
        table_b3.e[0]
        + table_b3.e[1] * np.log(rho) ** 2
        + table_b3.e[2] * np.sqrt(rho)
    )
    f = -np.exp(
        table_b3.f[0]
        + table_b3.f[1] * np.log(rho) ** 2
        + table_b3.f[2] * np.log(rho)
    )
    g = (
        table_b3.g[0]
        + table_b3.g[1] / np.log(rho)
        + table_b3.g[2] * np.log(rho) / rho
    ) ** -1
    coeffs = (jnp.float32(x) for x in (a, b, c, e, f, g))
    return IceVelocity(*coeffs)

  @classmethod
  def _precompute_snow_coeffs(cls, snow: Snow) -> SnowVelocity:
    """Precomputes the coefficient formulas from table B5 of Chen et al."""
    rho = snow.params.rho
    table_b5 = TableB5Coeffs()
    a = (
        table_b5.a[0]
        + table_b5.a[1] * np.log(rho)
        + table_b5.a[2] * rho ** (-3 / 2)
    )
    b = np.exp(
        table_b5.b[0]
        + table_b5.b[1] * np.log(rho) ** 2
        + table_b5.b[2] * np.log(rho)
    )
    c = np.exp(
        table_b5.c[0] + table_b5.c[1] / np.log(rho) + table_b5.c[2] / rho
    )
    e = (
        table_b5.e[0]
        + table_b5.e[1] * np.log(rho) * np.sqrt(rho)
        + table_b5.e[2] * np.sqrt(rho)
    )
    f = (
        table_b5.f[0]
        + table_b5.f[1] * np.log(rho)
        + table_b5.f[2] * np.exp(table_b5.f[3] - rho)
    )
    g = (
        table_b5.g[0]
        + table_b5.g[1] * np.log(rho) * np.sqrt(rho)
        + table_b5.g[2] / np.sqrt(rho)
    ) ** -1
    h = (
        table_b5.h[0]
        + table_b5.h[1] * rho ** (5.0 / 2.0)
        + table_b5.h[2] * np.exp(table_b5.h[3] - rho)
    )
    coeffs = (jnp.float32(x) for x in (a, b, c, e, f, g, h))
    return SnowVelocity(*coeffs)

  @classmethod
  def from_config(
      cls,
      one_moment_params: microphysics_pb2.OneMoment,
  ) -> 'TerminalVelocityChen2022':
    """Creates an instance of TerminalVelocityChen2022 from config proto."""
    rain = Rain.from_config(one_moment_params.rain)
    snow = Snow.from_config(one_moment_params.snow)
    ice = Ice.from_config(one_moment_params.ice)
    ice_velocity = cls._precompute_ice_coeffs(ice)
    snow_velocity = cls._precompute_snow_coeffs(snow)
    return cls(
        _rain=rain,
        _snow=snow,
        _ice=ice,
        rain_velocity=TableB1Coeffs(),
        ice_velocity=ice_velocity,
        snow_velocity=snow_velocity,
    )

  def _convert_coefficients_to_si_units_and_wrap(
      self,
      a: tuple[ScalarField, ...],
      b: tuple[ScalarField, ...],
      c: tuple[ScalarField, ...],
  ) -> TerminalVelocityCoefficients:
    """Converts the coefficients to SI units and puts them in a dataclass."""
    # Convert a from mm^-b to m^-b.
    a = tuple(a_i * jnp.power(1e3, b_i) for a_i, b_i in zip(a, b))
    # Convert c from mm^-1 to m^-1.
    c = tuple(c_i * 1e3 for c_i in c)
    return TerminalVelocityCoefficients(a, b, c)

  def _compute_raindrop_coefficients(
      self, rho: ScalarField
  ) -> TerminalVelocityCoefficients:
    """Computes raindrop coefficients as a function of density (table b1)."""
    coeffs = self.rain_velocity
    q = jnp.exp(coeffs.q_coeff * rho)
    a = (
        coeffs.a[0] * q,
        coeffs.a[1] * q,
        coeffs.a[2] * q * jnp.power(rho, coeffs.rho_exp),
    )
    b = tuple(coeffs.b[i][0] + coeffs.b[i][1] * rho for i in range(3))
    c = tuple(jnp.float32(c_i) for c_i in coeffs.c)
    return self._convert_coefficients_to_si_units_and_wrap(a, b, c)

  def _compute_ice_coefficients(
      self, rho: ScalarField
  ) -> TerminalVelocityCoefficients:
    """Computes the ice coefficients as a function of density."""
    coeffs = self.ice_velocity
    a = (
        coeffs.e * rho**coeffs.a,
        coeffs.f * rho**coeffs.a,
    )
    b = (
        coeffs.b + coeffs.c * rho,
        coeffs.b + coeffs.c * rho,
    )
    c = (jnp.zeros_like(coeffs.g), coeffs.g)
    return self._convert_coefficients_to_si_units_and_wrap(a, b, c)

  def _compute_snow_coefficients(
      self, rho: ScalarField
  ) -> TerminalVelocityCoefficients:
    """Computes the snow coefficients as a function of density."""
    coeffs = self.snow_velocity
    a = (
        coeffs.b * rho**coeffs.a,
        coeffs.e * rho**coeffs.a * jnp.exp(coeffs.h * rho),
    )
    b = (coeffs.c, coeffs.f)
    c = (jnp.zeros_like(coeffs.g), coeffs.g)
    return self._convert_coefficients_to_si_units_and_wrap(a, b, c)

  def _fall_speed_gamma_type(
      self,
      coeffs: TerminalVelocityCoefficients,
      lam: ScalarField,
  ) -> ScalarField:
    """Computes the gamma-type mass-weighted bulk fall speed formula.

    The equation is given by equation 20 of Chen et al. (2022).

    Args:
      coeffs: Terminal velocity coefficients a, b, c.
      lam: The Marshall-Palmer distribution rate parameter lambda.

    Returns:
      The mass-weighted bulk fall speed for a particle group.
    """
    mu = 0
    k = 3
    delta = mu + k + 1.0

    def compute_addend(
        a: ScalarField, b: ScalarField, c: ScalarField
    ) -> ScalarField:
      """Returns addend of the bulk fall speed for given coefficients."""
      lambda_independent_factor = jnp.where(
          particles.gamma(delta) == 0,
          0.0,
          a * particles.gamma(b + delta) / particles.gamma(delta),
      )
      lambda_ratio = jnp.where(
          lam == 0, 0.0, 1.0 / (1.0 + jnp.where(lam == 0, 0.0, c / lam))
      )
      lambda_dependent_factor = jnp.where(
          (lam + c) == 0,
          0.0,
          lambda_ratio**delta / (lam + c) ** b,
      )
      return lambda_independent_factor * lambda_dependent_factor

    result = jnp.array(0.0)
    for a_i, b_i, c_i in zip(coeffs.a, coeffs.b, coeffs.c):
      result = result + compute_addend(a_i, b_i, c_i)
    return result

  def _fall_speed_gamma_type_individual(
      self,
      coeffs: TerminalVelocityCoefficients,
      diameter: ScalarField,
  ) -> ScalarField:
    """Computes the terminal velocity of a single particle.

    This evaluates equation 19 of Chen et al. (2022) excluding the aspect
    ratio factor.

    Args:
      coeffs: Terminal velocity coefficients a, b, c.
      diameter: Pointwise estimated group droplet diameter.

    Returns:
      The mass-weighted bulk fall speed for a particle group.
    """
    result = jnp.array(0.0)
    for a, b, c in zip(coeffs.a, coeffs.b, coeffs.c):
      result = result + a * jnp.power(diameter, b) * jnp.exp(-c * diameter)
    return result

  def rain_terminal_velocity(
      self,
      rho: ScalarField,
      q_r: ScalarField,
  ) -> ScalarField:
    """Computes the terminal velocity of raindrops.

    Args:
      rho: The density of air [kg/m^3].
      q_r: The rain mass fraction [kg/kg].

    Returns:
      The terminal velocity of rain drops [m/s].
    """
    lam = particles.marshall_palmer_distribution_parameter_lambda(
        self._rain, rho, q_r
    )
    coeffs = self._compute_raindrop_coefficients(rho)
    return jnp.maximum(self._fall_speed_gamma_type(coeffs, lam), 0.0)

  def snow_terminal_velocity(
      self,
      rho: ScalarField,
      q_s: ScalarField,
  ) -> ScalarField:
    """Computes the terminal velocity of snow.

    Args:
      rho: The density of air [kg/m^3].
      q_s: The snow mass fraction [kg/kg].

    Returns:
      The terminal velocity of snow flakes [m/s].
    """
    lam = particles.marshall_palmer_distribution_parameter_lambda(
        self._snow, rho, q_s
    )
    psi_avg = jnp.where(
        lam == 0, 0.0, self._snow.phi_0 / jnp.power(lam, self._snow.alpha)
    )
    coeffs = self._compute_snow_coefficients(rho)
    fall_speed = jnp.power(psi_avg, self._KAPPA) * self._fall_speed_gamma_type(
        coeffs, lam
    )
    return jnp.maximum(fall_speed, 0.0)

  def condensate_terminal_velocity(
      self,
      particle: Rain | Ice,
      rho: ScalarField,
      q_sed: ScalarField,
  ) -> ScalarField:
    """Computes the sedimentation terminal velocity of cloud droplets or ice.

    Args:
      particle: Dataclass object storing constant parameters for the sediment.
      rho: The density of air [kg/m^3].
      q_sed: The sediment mass fraction [kg/kg].

    Returns:
      The terminal velocity of the sediment [m/s].
    """
    if isinstance(particle, Rain):
      coeffs = self._compute_raindrop_coefficients(rho)
      correction_factor = self._CLOUD_DROPLET_CORRECTION_FACTOR
    elif isinstance(particle, Ice):
      coeffs = self._compute_ice_coefficients(rho)
      correction_factor = 1.0
    else:
      raise ValueError(
          f'Sediment must be of type Rain or Ice, but got {type(particle)}.'
      )
    q_sed = jnp.maximum(q_sed, 0.0)
    diameter = jnp.power(
        rho * q_sed / constants.DROPLET_N / particle.params.rho, 1.0 / 3.0
    )
    fall_speed = self._fall_speed_gamma_type_individual(coeffs, diameter)
    return jnp.maximum(correction_factor * fall_speed, 0.0)
