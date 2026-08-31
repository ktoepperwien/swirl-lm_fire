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

# Copyright 2024 Google LLC
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
"""Defines an abstract class for the turbulent-combustion models (JAX).

This is the JAX port of
`swirl_lm.physics.turbulent_combustion.turbulent_combustion_generic`.
"""


import abc

from swirl_lm.jax.utility import types
from swirl_lm.physics.turbulent_combustion import turbulent_combustion_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


class TurbulentCombustionGeneric(abc.ABC):
  """Defines an abstract class for the turbulent-combustion models."""

  def __init__(
      self, model_params: turbulent_combustion_pb2.TurbulentCombustion
  ):
    """Initializes the turbulent combustion model."""
    self.model_params = model_params

  def update_diffusivity(
      self,
      diffusivity: ScalarField,
      states: ScalarFieldMap | None = None,
      additional_states: ScalarFieldMap | None = None,
  ) -> ScalarField:
    """Updates the diffusivity if provided by the turbulent combustion model."""
    del states, additional_states  # unused.
    return diffusivity

  def update_source_term(
      self,
      reaction_rate: ScalarField,
      states: ScalarFieldMap | None = None,
      additional_states: ScalarFieldMap | None = None,
  ) -> ScalarField:
    """Updates the reaction rate with the turbulence closure model."""
    del states, additional_states  # unused.
    return reaction_rate
