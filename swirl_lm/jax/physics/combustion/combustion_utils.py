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
"""Utility functions for combustion modeling (JAX port).

JAX port of standalone utility functions from
`swirl_lm.physics.combustion.combustion`.
"""


from collections.abc import Callable
import dataclasses

from swirl_lm.jax.utility import types

ScalarField = types.ScalarField


@dataclasses.dataclass
class IgnitionWithHotKernel:
  """Defines parameters for the high-temperature ignition kernel."""

  ignition_temperature: float = 800.0


@dataclasses.dataclass
class IgnitionWithHeatSource:
  """Defines parameters for heat-source ignition."""

  heat_source_magnitude: float = 50.0
  t_0: float = 0.0
  t_1: float = 300.0
  t_2: float = 2100.0
  t_3: float = 2400.0


def ramp_up_down_function(
    t_0: float,
    t_1: float,
    t_2: float,
    t_3: float,
) -> Callable[[float], float]:
  """Returns the value of the ramp up/down function at time `t`.

  The function is defined as:
    t <= t_0: 0
    t_0 < t <= t_1: linearly ramp up from 0 to 1
    t_1 < t <= t_2: 1
    t_2 < t <= t_3: linearly ramp down from 1 to 0
    t > t_3: 0

  Args:
    t_0: Time at the start of the ramp up.
    t_1: Time at the end of the ramp up.
    t_2: Time at the start of the ramp down.
    t_3: Time at the end of the ramp down.

  Returns:
    A function that takes time as the input, and returns a scaling factor.
  """
  assert t_0 <= t_1 <= t_2 <= t_3, (
      't_0, t_1, t_2, and t_3 must be in ascending order, but '
      f'got {t_0}, {t_1}, {t_2}, {t_3}'
  )

  def scaling_factor(t: float) -> float:
    """Computes the scaling factor following the ramp up and down function."""
    if t <= t_0 or t > t_3:
      return 0.0
    elif t <= t_1:
      return (t - t_0) / (t_1 - t_0) if t_1 > t_0 else 1.0
    elif t <= t_2:
      return 1.0
    else:
      return (t - t_3) / (t_2 - t_3)

  return scaling_factor


def ignition_with_heat_source(
    heat_source_magnitude: float,
    heat_source_coeff_fn: Callable[[float], float],
) -> Callable[[ScalarField, float], ScalarField]:
  """Generates a function that produces a heat source for ignition.

  The shape of the source term is prescribed by a helper variable named
  `ignition_kernel`. The magnitude is the product of `heat_source_magnitude`
  and the scaling factor obtained from `heat_source_coeff_fn`.

  Args:
    heat_source_magnitude: The maximum value of the heat source.
    heat_source_coeff_fn: A function that computes a scaling factor at time t.

  Returns:
    A function that takes a 3D ignition kernel array and time, returns a heat
    source for ignition.
  """

  def heat_source_update_fn(
      ignition_kernel: ScalarField,
      t: float,
  ) -> ScalarField:
    """Computes the heat source at `t` given the ignition kernel."""
    coeff = heat_source_coeff_fn(t)
    return coeff * heat_source_magnitude * ignition_kernel

  return heat_source_update_fn
