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
"""Utilities for working with Lagrangian particles (JAX port).

JAX port of `swirl_lm.physics.lpt.lpt_utils`.
"""



from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.physics.lpt import lpt_types
from swirl_lm.jax.utility import types

ScalarFieldMap = types.ScalarFieldMap


def get_particle_replica_id(
    locations: jax.Array,
    core_spacings: jax.Array,
    replicas: np.ndarray,
    global_min_pt: jax.Array,
) -> jax.Array:
  """A fast determination of the replicas containing particles on a uniform grid.

  Returns -1 if the particle is out of bounds. For the below definitions,
  x0, x1, x2 correspond to the z, x, and y dimensions respectively.

  Args:
    locations: A tensor of shape (n, 3) containing the z, x, y particle
      locations.
    core_spacings: A three element tensor of floats z, x, y representing the
      length of each dimension contained within a replica's partial domain.
    replicas: A 3D numpy array of core replica IDs of shape cx, cy, cz.
    global_min_pt: A size three tuple containing the global minimum point.

  Returns:
    A tensor of replica IDs of the particles, -1 if out of bounds.
  """
  x0_ind, x1_ind, x2_ind = [
      jnp.floor((locations[:, i] - global_min_pt[i]) / core_spacings[i]).astype(
          lpt_types.LPT_INT
      )
      for i in range(3)
  ]

  core_n1, core_n2, core_n0 = replicas.shape

  out_of_bounds = (
      (x0_ind < 0)
      | (x0_ind >= core_n0)
      | (x1_ind < 0)
      | (x1_ind >= core_n1)
      | (x2_ind < 0)
      | (x2_ind >= core_n2)
  )

  replicas_arr = jnp.array(replicas, dtype=lpt_types.LPT_INT)

  # Gather from replicas using (x1, x2, x0) indices.
  # Clip indices to valid range for gathering; out-of-bounds handled by mask.
  x0_clamped = jnp.clip(x0_ind, 0, core_n0 - 1)
  x1_clamped = jnp.clip(x1_ind, 0, core_n1 - 1)
  x2_clamped = jnp.clip(x2_ind, 0, core_n2 - 1)
  gathered = replicas_arr[x1_clamped, x2_clamped, x0_clamped]

  replica_ids = jnp.where(out_of_bounds, -jnp.ones_like(x0_ind), gathered)
  return replica_ids


def fluid_data_linear_interpolation(
    locations: jax.Array,
    states: ScalarFieldMap,
    variables: Sequence[str],
    grid_spacings: jax.Array,
    local_grid_min_pt: jax.Array,
) -> jax.Array:
  """Interpolates local fluid data within the replica.

  Calculates fluid variables for the particles in the local replica assuming a
  uniform grid. The variables are trilinearly interpolated from the fluid grid
  points.

  Args:
    locations: An (n, 3) array containing `n` locations at (z, x, y).
    states: A ScalarFieldMap containing the fluid states.
    variables: Names of the fluid variables to interpolate.
    grid_spacings: The grid spacings in the z, x, and y dimensions.
    local_grid_min_pt: The minimum point of the local grid including halos.

  Returns:
    An (n, m) array containing fluid data at `n` locations for `m` variables.
  """
  # Stack all requested fields into a single (nx, ny, nz, m) array.
  field_data = jnp.stack([states[v] for v in variables], axis=-1)

  # Compute fractional indices.
  frac_indices = (locations - local_grid_min_pt) / grid_spacings

  # Integer lower-corner indices.
  idx = jnp.floor(frac_indices).astype(jnp.int32)
  # Weights for upper corner.
  w = frac_indices - idx

  # Trilinear interpolation.
  result = jnp.zeros((locations.shape[0], len(variables)))
  for dz in (0, 1):
    for dx in (0, 1):
      for dy in (0, 1):
        weight = (
            ((1 - w[:, 0]) if dz == 0 else w[:, 0])
            * ((1 - w[:, 1]) if dx == 0 else w[:, 1])
            * ((1 - w[:, 2]) if dy == 0 else w[:, 2])
        )
        corner_vals = field_data[idx[:, 0] + dz, idx[:, 1] + dx, idx[:, 2] + dy]
        result = result + weight[:, None] * corner_vals

  return result


def get_active_rows(
    lpt_field_ints: lpt_types.LptFieldInts,
    float_field: lpt_types.LptFieldFloats,
) -> tuple[lpt_types.LptFieldFloats, jax.Array]:
  """Returns only the rows corresponding to the active particles.

  Args:
    lpt_field_ints: A (n, 2) array of integer particle states.
    float_field: A (n, m) array of float particle data.

  Returns:
    A (q, m) array of float particle states of only active particles.
    A (q,) array of indices of the active rows.
  """
  statuses = lpt_field_ints[:, 0]
  active_indices = jnp.where(statuses == 1, size=float_field.shape[0])[0]
  # Count actual active particles.
  active_tensor = jax.nn.one_hot(active_indices, float_field.shape[0])
  active_rows = jnp.einsum('qn,nm->qm', active_tensor, float_field)
  return active_rows, active_indices


def tensor_scatter_update(
    tensor: jax.Array,
    indices: jax.Array,
    updates: jax.Array,
) -> jax.Array:
  """Updates the rows in a 2D array with updates at the given indices.

  Args:
    tensor: A (n, m) array that will be updated.
    indices: A (q,) array of indices of the rows to update.
    updates: A (q, m) array of rows that will overwrite rows in `tensor`.

  Returns:
    Array with `updates` applied at the rows denoted by `indices`.
  """
  one_hot = jax.nn.one_hot(indices, tensor.shape[0], dtype=lpt_types.LPT_FLOAT)
  substitute = jnp.einsum('qj,qi->ij', updates, one_hot).astype(
      lpt_types.LPT_FLOAT
  )
  inverted = jnp.where(
      jnp.abs(substitute) > np.finfo(np.float32).resolution,
      jnp.zeros_like(substitute),
      jnp.ones_like(substitute),
  ).astype(lpt_types.LPT_FLOAT)
  return tensor * inverted + substitute
