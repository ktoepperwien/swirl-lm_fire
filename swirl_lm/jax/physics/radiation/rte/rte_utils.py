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

"""Utility library for solving the radiative transfer equation (RTE) (JAX).

JAX port of `swirl_lm.physics.radiation.rte.rte_utils`.
"""


import inspect
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

X0_KEY = 'x0'
PRIMARY_GRID_KEY = 'primary'
EXTENDED_GRID_KEY = 'extended'


class RTEUtils:
  """A library for distributing radiative transfer computations on devices.

  Attributes:
    params: An instance of `GridParametrization` containing the grid dimensions
      and information about the computational topology.
    num_cores: A 3-tuple containing the number of cores assigned to each
      dimension.
    grid_size: The local grid dimensions per core.
    halos: The number of halo points on each face of the grid.
  """

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
  ):
    self.params = params
    self.num_cores = (params.cx, params.cy, params.cz)
    self.grid_size = (params.nx, params.ny, params.nz)
    self.halos = params.halo_width

  def _append(
      self,
      a: ScalarField,
      b: ScalarField,
      dim: int,
      forward: bool = True,
  ) -> ScalarField:
    """Appends `a` to `b` along `dim` if `forward`, `b` to `a` otherwise."""
    if not forward:
      a, b = b, a
    return jnp.concatenate([a, b], axis=dim)

  def _pad(
      self,
      f: ScalarField,
      low_n: int,
      high_n: int,
      dim: int,
  ) -> ScalarField:
    """Pads the field with zeros along the dimension `dim`."""
    paddings = [(0, 0)] * 3
    paddings[dim] = (low_n, high_n)
    return jnp.pad(f, paddings)

  def _generate_adjacent_pair_assignments(
      self, replicas: np.ndarray, axis: int, forward: bool
  ) -> list[np.ndarray]:
    """Creates groups of source-target device pairs along `axis`.

    The group assignments are used by `jax.lax.ppermute` to exchange data
    between neighboring devices along `axis`.

    Args:
      replicas: The mapping from the core coordinate to the local replica id.
      axis: The axis along which data will be propagated.
      forward: Whether the data propagation will unravel in the direction of
        increasing index along `axis`.

    Returns:
      A list of groups of adjacent replica id pairs. There will be one such
      group for every interface of the computational topology along `axis`.
    """
    groups = _group_replicas(replicas, axis=axis)
    pair_groups = []
    depth = groups.shape[1]
    for i in range(depth - 1):
      pair_group = groups[:, i : i + 2]
      if not forward:
        # Reverse the order of the source-target pairs.
        pair_group = pair_group[:, ::-1]
      pair_groups.append(pair_group.tolist())
    return pair_groups

  def _local_recurrent_op(
      self,
      recurrent_fn: Callable[..., jax.Array],
      variables: ScalarFieldMap,
      dim: int,
      n: int,
      forward: bool = True,
  ) -> tuple[ScalarField, ScalarField]:
    """Computes a sequence of recurrent operations along a dimension.

    Each core performs the same operation on data local to it independently.
    Note that the initial input `x0` in `variables` is not included in the final
    output.

    Args:
      recurrent_fn: The local cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`, which should correspond
        to the boundary solution that initiates the recurrence.
      dim: The physical dimension along which the sequence will be applied.
      n: The number of layers in the final solution.
      forward: Whether the accumulation starts with the first layer.

    Returns:
      A tuple containing 1) A 3D variable with the cumulative output from the
      chain of recurrent transformations. 2) the 2D output of the last
      recurrent transformation.
    """
    x = variables[X0_KEY]

    for i in range(n):
      prev_idx = i - 1
      slice_idx = i if forward else -i - 1
      plane_args = {
          k: common_ops.slice_field(v, dim, slice_idx, size=1)
          for k, v in variables.items()
          if k != X0_KEY
      }
      prev_slice_idx = prev_idx if forward else -prev_idx - 1
      plane_args[X0_KEY] = (
          x
          if i == 0
          else common_ops.slice_field(x, dim, prev_slice_idx, size=1)
      )
      arg_lst = [
          plane_args[k] for k in inspect.getfullargspec(recurrent_fn).args
      ]
      next_layer = recurrent_fn(*arg_lst)
      x = next_layer if i == 0 else self._append(x, next_layer, dim, forward)

    last_layer = -1 if forward else 0
    last_local_layer = common_ops.slice_field(x, dim, last_layer, size=1)

    return x, last_local_layer

  def _cumulative_recurrent_op_sequential(
      self,
      mesh: Mesh,  # pylint: disable=unused-argument
      grid_params: grid_parametrization.GridParametrization,
      replicas: np.ndarray,
      recurrent_fn: Callable[..., ScalarField],
      variables: ScalarFieldMap,
      dim: int,
      forward: bool = True,
  ) -> ScalarField:
    """Computes a sequence of recurrent operations globally.

    This particular implementation is sequential, so every layer of devices
    along the accumulation axis needs to wait for the previous computational
    layer to complete before proceeding.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      replicas: The mapping from the core coordinate to the local replica id.
      recurrent_fn: The local cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`.
      dim: The physical dimension along which the recurrence will be applied.
      forward: Whether the accumulation starts with the first layer.

    Returns:
      A 3D variable with the cumulative output from the chain of recurrent
      transformations.
    """
    halo_widths = [0, 0, 0]
    halo_widths[dim] = self.halos

    n = self.grid_size[dim] - 2 * self.halos

    # Remove halos along the axis.
    kwargs: dict[str, ScalarField] = {
        k: common_ops.strip_halos(
            v,
            halo_widths[0],
            halo_widths[1],
            halo_widths[2],
            grid_params,
        )
        for k, v in variables.items()
        if k != X0_KEY
    }
    kwargs[X0_KEY] = variables[X0_KEY]

    def local_fn(
        x0: ScalarField,
    ) -> tuple[ScalarField, ScalarField]:
      """Generates the output of a cumulative operation and its last layer."""
      kwargs[X0_KEY] = x0
      return self._local_recurrent_op(recurrent_fn, kwargs, dim, n, forward)

    core_idx = _get_core_coordinate(replicas, dim)

    # Cumulative local output and its last layer.
    x_cum, x_out = local_fn(x0=variables[X0_KEY])

    pair_groups = self._generate_adjacent_pair_assignments(
        replicas, dim, forward
    )
    n_groups = len(pair_groups)
    interface_iter = range(n_groups) if forward else reversed(range(n_groups))

    # Sequentially evaluate a level and propagate results to the next level.
    for i in interface_iter:
      pair_group = pair_groups[i]
      # Send / receive the last recurrent output layer.
      x_prev = jax.lax.ppermute(
          x_out,
          axis_name=('x', 'y', 'z')[dim],
          perm=pair_group,
      )
      # Index of the next set of cores receiving the data.
      recv_core_idx = i + 1 if forward else i
      x_cum, x_out = jax.lax.cond(
          core_idx == recv_core_idx,
          lambda: local_fn(x0=x_prev),  # pylint: disable=cell-var-from-loop
          lambda: (x_cum, x_out),
      )

    # Pad the result with halo layers.
    return self._pad(x_cum, self.halos, self.halos, dim)

  def _exchange_halos(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      f: ScalarField,
      x0: ScalarField,
      dim: int,
      x0_face: Literal[0, 1],
  ) -> ScalarField:
    """Exchanges halos along the specified dimension."""
    bc: list[  # pyrefly: ignore[type-annotation]
        tuple[tuple[halo_exchange.BCType, float | list[jax.Array]] | None, ...]
        | None
    ] = [
        (
            (halo_exchange.BCType.NEUMANN, 0.0),
            (halo_exchange.BCType.NEUMANN, 0.0),
        )
        for _ in range(3)
    ]
    # Set the boundary plane that initiates the recurrent operation as the
    # boundary values.
    dim_bc = list(bc[dim])  # pyrefly: ignore[code]
    dim_bc[x0_face] = (
        halo_exchange.BCType.DIRICHLET,
        [x0] * self.halos,
    )
    bc[dim] = tuple(dim_bc)  # pyrefly: ignore[code]
    return halo_exchange.inplace_halo_exchange(
        f,
        axes=('x', 'y', 'z'),
        mesh=mesh,
        grid_params=grid_params,
        boundary_conditions=tuple(bc),  # pyrefly: ignore[code]
        halo_width=self.halos,
    )

  def cumulative_recurrent_op(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      replicas: np.ndarray,
      recurrent_fn: Callable[..., ScalarField],
      variables: ScalarFieldMap,
      dim: int,
      forward: bool = True,
  ) -> dict[str, ScalarField]:
    """Applies a recurrent operation globally along a specified dimension.

    This global operation is sequential and will process each layer along `dim`
    at a time.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      replicas: The mapping from the core coordinate to the local replica id.
      recurrent_fn: The cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`.
      dim: The physical dimension along which the recurrence will be applied.
      forward: Whether the accumulation starts with the first layer.

    Returns:
      A dictionary containing a single entry ('primary') for the 3D field with
      the cumulative output.
    """
    # Store the initial plane to be used as the boundary value.
    x0 = variables[X0_KEY]
    # Face that initiates the recurrence.
    x0_face = 0 if forward else 1

    val = self._cumulative_recurrent_op_sequential(
        mesh, grid_params, replicas, recurrent_fn, variables, dim, forward
    )
    return {
        PRIMARY_GRID_KEY: self._exchange_halos(
            mesh, grid_params, val, x0, dim, x0_face
        )
    }


def _group_replicas(
    replicas: np.ndarray,
    axis: int,
) -> np.ndarray:
  """Groups replica ids along the specified axis.

  Args:
    replicas: A 3D numpy array mapping core coordinates to replica ids.
    axis: The axis along which to group.

  Returns:
    A 2D numpy array where each row contains the replica ids along `axis`
    for a specific combination of the other two axes.
  """
  # Move the target axis to position 1 and flatten the other axes.
  shape = replicas.shape
  other_axes = [i for i in range(3) if i != axis]
  n_groups = shape[other_axes[0]] * shape[other_axes[1]]
  depth = shape[axis]
  groups = np.moveaxis(replicas, axis, -1).reshape(n_groups, depth)
  return groups


def _get_core_coordinate(
    replicas: np.ndarray,
    dim: int,
) -> jax.Array:
  """Gets the coordinate of the current device along `dim`.

  In JAX SPMD, the axis index is obtained from the mesh. This function
  provides a placeholder that uses `jax.lax.axis_index`.

  Args:
    replicas: The mapping from the core coordinate to the local replica id.
    dim: The dimension to get the coordinate for.

  Returns:
    The index of the current device along `dim`.
  """
  del replicas  # Unused.
  axis_name = ('x', 'y', 'z')[dim]
  return jax.lax.axis_index(axis_name)
