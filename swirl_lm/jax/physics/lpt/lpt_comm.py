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

"""Communication library for the lagrangian particles (JAX port).

JAX port of `swirl_lm.physics.lpt.lpt_comm`.
"""


from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.physics.lpt import lpt_types
from swirl_lm.jax.physics.lpt import lpt_utils
from swirl_lm.jax.utility import types

ScalarFieldMap = types.ScalarFieldMap


def _send_recv(
    data: jax.Array,
    source_dest_pairs: np.ndarray,
    n_max: int,
) -> jax.Array:
  """Exchanges N-D arrays across a list of (sender, receiver) pairs.

  This is a JAX equivalent of `swirl_lm.communication.send_recv.send_recv`,
  using `jax.lax.ppermute` instead of `tf.raw_ops.CollectivePermute`.

  Args:
    data: The n-dimensional array to be sent. Dimension 0 can differ across
      replicas, but is padded to `n_max` for communication.
    source_dest_pairs: A 2-D numpy array of shape `[num_replicas, 2]` with
      sender-receiver pairs.
    n_max: Buffer size for communication.

  Returns:
    An N-D array received from the sender replica.
  """
  # Pad data to n_max along dimension 0.
  pad_width = [(0, n_max - data.shape[0])] + [(0, 0)] * (data.ndim - 1)
  padded = jnp.pad(data, pad_width)

  n_sent = jnp.int32(data.shape[0])

  # Exchange size and data.
  pairs = [tuple(row) for row in source_dest_pairs.tolist()]
  n_received = jax.lax.ppermute(n_sent, axis_name='i', perm=pairs)
  received = jax.lax.ppermute(padded, axis_name='i', perm=pairs)

  # Trim to the original size from the sender.
  return received[:n_received]


def pairwise(
    locs: jax.Array,
    states: ScalarFieldMap,
    replica_id: jax.Array,
    replicas: np.ndarray,
    variables: Sequence[str],
    grid_spacings: jax.Array,
    core_spacings: jax.Array,
    local_min_pt: jax.Array,
    global_min_pt: jax.Array,
    n_max: int,
) -> jax.Array:
  """Requests fluid data for particles located on other replicas.

  This function conducts n^2 exchanges for n replicas. Each replica sends and
  receives fluid data from all other replicas via circular exchange patterns.

  Args:
    locs: An (n, 3) float array of physical locations in z, x, y order.
    states: A dictionary containing the fluid data.
    replica_id: The replica id of the local replica.
    replicas: A numpy array of shape `(cx, cy, cz)` with replica ids.
    variables: Variables to be exchanged (e.g., ["w", "u", "v"]).
    grid_spacings: Grid spacings in z, x, y order.
    core_spacings: Core partial domain sizes in z, x, y.
    local_min_pt: Minimum point of the local grid in z, x, y including halos.
    global_min_pt: Global minimum point in z, x, y.
    n_max: Maximum number of elements any replica can send to another.

  Returns:
    A (n, len(variables)) array of fluid data from remote locations.
  """
  loc_replica_ids = lpt_utils.get_particle_replica_id(
      locs, core_spacings, replicas, global_min_pt
  )

  fluid_data = jnp.zeros(
      (locs.shape[0], len(variables)), dtype=lpt_types.LPT_FLOAT
  )

  num_replicas = replicas.size
  source = np.arange(num_replicas, dtype=lpt_types.LPT_NP_INT)

  for offset in range(num_replicas):
    dest = np.roll(source, offset, axis=0).astype(lpt_types.LPT_NP_INT)[::-1]
    source_dest_pair = np.stack([source, dest], axis=1)

    # Determine this replica's destination in the current exchange.
    dest_replica = jnp.take(dest, replica_id)

    # Gather locations of particles owned by this replica but on dest replica.
    dest_mask = loc_replica_ids == dest_replica
    dest_replica_loc_indices = jnp.nonzero(
        dest_mask, size=n_max, fill_value=-1
    )[0]
    dest_replica_loc = (
        jax.nn.one_hot(dest_replica_loc_indices, locs.shape[0]) @ locs
    )

    # Send particle locations to dest; receive locations from source.
    pairs = [tuple(row) for row in source_dest_pair.tolist()]
    dest_locations = jax.lax.ppermute(
        dest_replica_loc, axis_name='i', perm=pairs
    )

    # Interpolate at received locations.
    fluid_data_at_dest_locations = lpt_utils.fluid_data_linear_interpolation(
        dest_locations, states, variables, grid_spacings, local_min_pt
    )

    # Send interpolated data back; receive our data.
    replica_fluid_data = jax.lax.ppermute(
        fluid_data_at_dest_locations, axis_name='i', perm=pairs
    )

    # Update fluid_data at the corresponding indices.
    fluid_data = lpt_utils.tensor_scatter_update(
        fluid_data, dest_replica_loc_indices, replica_fluid_data
    )

  return fluid_data


def one_shuffle(
    locs: jax.Array,
    states: ScalarFieldMap,
    replica_id: jax.Array,
    replicas: np.ndarray,
    variables: Sequence[str],
    grid_spacings: jax.Array,
    core_spacings: jax.Array,
    local_min_pt: jax.Array,
    global_min_pt: jax.Array,
    n_max: int,
) -> jax.Array:
  """Circular replica communication to get particle fluid data.

  Each replica sends data to the next in a circle and receives from the
  previous. After n iterations (n = number of replicas), each replica has
  the fluid data for all of its particle locations.

  Args:
    locs: An (n, 3) float array of physical locations in z, x, y order.
    states: A `ScalarFieldMap` containing the fluid data.
    replica_id: The replica id of the local replica.
    replicas: A numpy array of shape `(cx, cy, cz)` with replica ids.
    variables: Variables to exchange (e.g., ["w", "u", "v"]).
    grid_spacings: Grid spacings in z, x, y order.
    core_spacings: Core partial domain sizes in z, x, y.
    local_min_pt: Minimum point of the local grid in z, x, y including halos.
    global_min_pt: Global minimum point in z, x, y excluding halos.
    n_max: Maximum elements any replica can send. Should be the number of rows
      in `lpt_field_` arrays.

  Returns:
    A (n, len(variables)) array of fluid data from remote locations.
  """
  num_replicas = replicas.size
  source = np.arange(num_replicas, dtype=lpt_types.LPT_NP_INT)
  dest = np.roll(source, 1, axis=0).astype(lpt_types.LPT_NP_INT)
  source_dest_pairs = [(int(s), int(d)) for s, d in zip(source, dest)]

  # Prepare joint location and field data.
  fluid_data = jnp.zeros((n_max, len(variables)), dtype=jnp.float32)
  loc_and_fluid_data = jnp.concatenate([locs, fluid_data], axis=1)

  for _ in range(num_replicas):
    # Circular exchange.
    loc_and_fluid_data = jax.lax.ppermute(
        loc_and_fluid_data, axis_name='i', perm=source_dest_pairs
    )

    current_locs = loc_and_fluid_data[:, :3]
    current_fluid_data = loc_and_fluid_data[:, 3:]

    # Find particles located on this replica.
    loc_replicas = lpt_utils.get_particle_replica_id(
        current_locs, core_spacings, replicas, global_min_pt
    )
    loc_indices_local = jnp.nonzero(
        loc_replicas == replica_id, size=n_max, fill_value=-1
    )[0]
    locs_local = jnp.einsum(
        'qj,ji->qi',
        jax.nn.one_hot(loc_indices_local, n_max),
        current_locs,
    )

    # Interpolate at local locations.
    fluid_data_at_locs = lpt_utils.fluid_data_linear_interpolation(
        locs_local, states, variables, grid_spacings, local_min_pt
    )

    # Update fluid data.
    current_fluid_data = lpt_utils.tensor_scatter_update(
        current_fluid_data, loc_indices_local, fluid_data_at_locs
    )

    # Reassemble joint tensor.
    loc_and_fluid_data = jnp.concatenate(
        [current_locs, current_fluid_data], axis=1
    )

  return loc_and_fluid_data[:, 3:]
