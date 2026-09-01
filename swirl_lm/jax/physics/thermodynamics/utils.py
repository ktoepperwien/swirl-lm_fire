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
"""Utility functions for thermodynamic computations.

JAX port of `swirl_lm.physics.thermodynamics.thermodynamics_utils`.
"""

import jax.numpy as jnp
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# Physical constants.
R_UNIVERSAL = 8.3144598  # Universal gas constant, J/mol/K.
R_D = 287.0  # Gas constant for dry air, J/kg/K.
G = 9.81  # Gravitational acceleration, N/kg.
DRY_AIR_MOLECULAR_WEIGHT = 0.0289647  # kg/mol.

# Name of the inert species (ambient air).
INERT_SPECIES = 'ambient'


def regularize_scalar_bound(phi: ScalarField | float) -> ScalarField:
  """Enforces a bound of [0, 1] on the scalar `phi`."""
  return jnp.clip(phi, 0.0, 1.0)


def regularize_scalar_sum(
    phi: dict[str, ScalarField],
) -> dict[str, ScalarField]:
  """Rescales the scalars so that their sum at each point is 1."""
  sc_total = sum(phi.values())
  return {name: val / sc_total for name, val in phi.items()}


def compute_ambient_air_fraction(
    phi: dict[str, ScalarField],
) -> ScalarField:
  """Computes the mass fraction of ambient air (1 - sum of other scalars)."""
  y_ambient = 1.0 - sum(phi.values())
  return regularize_scalar_bound(y_ambient)


def compute_mixture_molecular_weight(
    molecular_weights: dict[str, float],
    mass_fractions: dict[str, ScalarField],
) -> ScalarField:
  """Computes the mixture molecular weight from species mass fractions.

  Uses the reciprocal mixing rule: 1/W_mix = sum(Y_i / W_i).

  Args:
    molecular_weights: Dict mapping species name to molecular weight (kg/mol).
    mass_fractions: Dict mapping species name to mass fraction field.

  Returns:
    The mixture molecular weight field.
  """
  w_mix_inv = jnp.zeros_like(list(mass_fractions.values())[0])
  for sc_name, w_sc in molecular_weights.items():
    w_mix_inv = w_mix_inv + mass_fractions[sc_name] / w_sc
  return 1.0 / w_mix_inv
