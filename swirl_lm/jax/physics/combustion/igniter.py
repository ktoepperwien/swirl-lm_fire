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
"""A library of ignition functions (JAX port).

JAX port of `swirl_lm.physics.combustion.igniter`.
"""


from collections.abc import Callable, Sequence

import jax.numpy as jnp
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField

# A function that takes (xx, yy, zz, lx, ly, lz) and returns a scalar field.
ValueFunction = Callable[..., ScalarField]


class Igniter:
  """A library that manages the ignition kernel.

  The ignition schedule is created based on the following assumptions:
  1. Ignition starts from a single point.
  2. Ignition progresses with constant speed in all directions along a sphere.
  3. The shape is determined by an externally-defined binary mask.
  """

  def __init__(
      self,
      ignition_speed: float,
      ignition_start_point: Sequence[float],
      ignition_duration: float,
      start_step_id: int,
      igniter_radius: float,
      dt: float,
  ):
    """Initializes the ignition scheduling object.

    Args:
      ignition_speed: The speed that the ignition kernel moves.
      ignition_start_point: The (x, y, z) coordinates of the starting point.
      ignition_duration: The duration of the ignition event, in seconds.
      start_step_id: The step id at which the ignition starts.
      igniter_radius: The radius (meters) of the ignition kernel.
      dt: The time step size of the simulation.
    """
    self._speed = ignition_speed
    self._origin = ignition_start_point
    self._start_step_id = float(start_step_id)
    self._dt = dt

    self._start_time = self._start_step_id * self._dt
    self._end_time = self._start_time + ignition_duration
    self._igniter_radius_in_time = igniter_radius / self._speed

  def ignition_schedule_init_fn(
      self,
      ignition_kernel_shape_fn: ValueFunction,
  ) -> ValueFunction:
    """Generates an init function that guides the ignition sequence.

    Args:
      ignition_kernel_shape_fn: A function that provides a binary kernel
        specifying the overall shape of the ignition kernel.

    Returns:
      A function that generates a kernel of time relative to the start of
      the simulation.
    """

    def init_fn(
        xx: ScalarField,
        yy: ScalarField,
        zz: ScalarField,
        lx: float,
        ly: float,
        lz: float,
        *args,
    ) -> ScalarField:
      """Initializes the ignition sequence tensor."""
      distance = jnp.sqrt(
          (xx - self._origin[0]) ** 2
          + (yy - self._origin[1]) ** 2
          + (zz - self._origin[2]) ** 2
      )
      ignition_time = distance / self._speed + self._start_time
      ignition_kernel = ignition_kernel_shape_fn(xx, yy, zz, lx, ly, lz, *args)
      return jnp.where(
          ignition_kernel > 0.0,
          ignition_time,
          -2.0 * self._igniter_radius_in_time * jnp.ones_like(ignition_time),
      )

    return init_fn

  def ignition_kernel(
      self,
      step_id: int,
      ignition_schedule: ScalarField,
  ) -> ScalarField:
    """Generates a binary ignition kernel at the present step.

    Args:
      step_id: The current step id.
      ignition_schedule: A 3D array of floats representing the ignition time at
        each point.

    Returns:
      A binary array where 1.0 represents the location of the ignition kernel.
    """
    t = jnp.float32(step_id) * self._dt

    kernel = jnp.where(
        (ignition_schedule >= t - self._igniter_radius_in_time)
        & (ignition_schedule <= t + self._igniter_radius_in_time),
        jnp.ones_like(ignition_schedule),
        jnp.zeros_like(ignition_schedule),
    )

    # Limit ignition to the time interval.
    kernel = jnp.where(
        (t >= self._start_time) & (t <= self._end_time),
        kernel,
        jnp.zeros_like(kernel),
    )
    return kernel
