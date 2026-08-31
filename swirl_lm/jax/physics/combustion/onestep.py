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
r"""A library for computing reaction source terms with a one-step mechanism.

JAX port of `swirl_lm.physics.combustion.onestep`.

The one-step chemistry model for gaseous phase reaction is represented as:
  F + O -> P,
where:
  F is the fuel,
  O is the oxidizer, and
  P is the reaction product.
The reaction source term is a function of the mass fractions of the fuel and
oxidizer, as well as the temperature, which takes the form of the Arrhenius law:
  w(F, O, T) = A[F]^a[O]^b exp(-Ea/R/T),
"""


import jax.numpy as jnp
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# The universal gas constant (J/mol/K).
R_UNIVERSAL = 8.3145
# Temperature bounds (K).
T_MIN = 273.0
T_MAX = 2500.0


def _arrhenius_law(
    c_f: ScalarField,
    c_o: ScalarField,
    temperature: ScalarField,
    a_cst: float,
    coeff_f: float,
    coeff_o: float,
    e_a: float,
) -> ScalarField:
  """Computes the Arrhenius law."""
  return (
      a_cst
      * jnp.power(c_f, coeff_f)
      * jnp.power(c_o, coeff_o)
      * jnp.exp(-e_a / R_UNIVERSAL / temperature)
  )


def _concentration(
    y_species: ScalarField,
    w_species: float,
    rho: ScalarField,
) -> ScalarField:
  """Computes the volume concentration of species."""
  return rho * y_species / w_species


def one_step_reaction_source(
    y_f: ScalarField,
    y_o: ScalarField,
    temperature: ScalarField,
    rho: ScalarField,
    a_cst: float,
    coeff_f: float,
    coeff_o: float,
    e_a: float,
    q: float,
    cp: float,
    w_f: float,
    w_o: float,
    nu_f: float = 1.0,
    nu_o: float = 1.0,
) -> list[ScalarField]:
  """Computes the reaction source term using onestep chemistry.

  Args:
    y_f: The mass fraction of fuel.
    y_o: The mass fraction of oxidizer.
    temperature: The temperature, in units of K.
    rho: The density of the flow field, in units of kg/m^3.
    a_cst: The constant A in the Arrhenius law.
    coeff_f: The power law coefficient of the fuel volume concentration.
    coeff_o: The power law coefficient of the oxidizer volume concentration.
    e_a: The activation energy.
    q: The heat of combustion.
    cp: The specific heat.
    w_f: The molecular weight of the fuel.
    w_o: The molecular weight of the oxidizer.
    nu_f: The stoichiometric coefficient of the fuel.
    nu_o: The stoichiometric coefficient of the oxidizer.

  Returns:
    The rate of change of the y_f, y_o, and temperature due to onestep reaction.
  """
  c_f = _concentration(jnp.clip(y_f, 0.0, 1.0), w_f, rho)
  c_o = _concentration(jnp.clip(y_o, 0.0, 1.0), w_o, rho)

  omega = _arrhenius_law(
      c_f,
      c_o,
      jnp.clip(temperature, T_MIN, T_MAX),
      a_cst,
      coeff_f,
      coeff_o,
      e_a,
  )

  return [
      -nu_f * w_f * omega / rho,
      -nu_o * w_o * omega / rho,
      q * omega / cp / rho,
  ]


def reaction_source_update_fn(
    a_cst: float,
    coeff_f: float,
    coeff_o: float,
    e_a: float,
    q: float,
    cp: float,
    w_f: float,
    w_o: float,
    nu_f: float = 1.0,
    nu_o: float = 1.0,
):
  """Generates an update function of reaction source terms and heat release.

  Args:
    a_cst: The constant A in the Arrhenius law.
    coeff_f: The power law coefficient of the fuel volume concentration.
    coeff_o: The power law coefficient of the oxidizer volume concentration.
    e_a: The activation energy.
    q: The heat of combustion.
    cp: The specific heat.
    w_f: The molecular weight of the fuel.
    w_o: The molecular weight of the oxidizer.
    nu_f: The stoichiometric coefficient of the fuel.
    nu_o: The stoichiometric coefficient of the oxidizer.

  Returns:
    A function that updates additional_states with src_Y_F, src_Y_O, src_T.
  """

  def additional_states_update_fn(
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarFieldMap:
    """Computes the reaction source term for Y_F, Y_O, and T."""
    source_terms = one_step_reaction_source(
        states['Y_F'],
        states['Y_O'],
        states['T'],
        states['rho'],
        a_cst,
        coeff_f,
        coeff_o,
        e_a,
        q,
        cp,
        w_f,
        w_o,
        nu_f,
        nu_o,
    )
    updated: dict[str, ScalarField] = {}
    for varname, value in additional_states.items():
      if varname == 'src_Y_F':
        updated[varname] = source_terms[0]
      elif varname == 'src_Y_O':
        updated[varname] = source_terms[1]
      elif varname == 'src_T':
        updated[varname] = source_terms[2]
      else:
        updated[varname] = value
    return updated

  return additional_states_update_fn
