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

"""A library of boundary condition related utility functions."""

from collections.abc import Sequence
import enum
from typing import Any, Literal, TypeAlias

from absl import logging
import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.jax.base import physical_variable_keys_manager
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types

BoundaryConditionDict: TypeAlias = dict[
    str, halo_exchange.BoundaryConditionsSpec | None
]

ScalarFieldMap: TypeAlias = types.ScalarFieldMap


class BoundaryType(enum.Enum):
  """Defines the physical type of the boundary."""

  UNKNOWN = 0
  SLIP_WALL = 1
  NON_SLIP_WALL = 2
  PERIODIC = 3
  INFLOW = 4
  OUTFLOW = 5
  SHEAR_WALL = 6


def find_bc_type(
    bc: BoundaryConditionDict, periodic_dims: list[bool]
) -> list[list[BoundaryType | None]]:
  """Finds the type of each boundary based on boundary conditions."""
  bc_type = [[None, None], [None, None], [None, None]]

  if bc['u'] is None or bc['v'] is None or bc['w'] is None:
    return [[BoundaryType.PERIODIC] * 2] * 3

  def velocity_var(dim: int):
    """The name of the velocity variable in the given dimension."""
    if dim not in range(3):
      raise ValueError(
          'Dimension has to be one of 0, 1, and 2. Given {}.'.format(dim)
      )
    return ('u', 'v', 'w')[dim]

  def is_non_slip_wall(dim: int, face: int):
    """Checks if the boundary is a non-slip wall."""
    return (
        (
            bc['u'][dim][face][0] == halo_exchange.BCType.DIRICHLET  # pyrefly: ignore[unsupported-operation]
            and bc['u'][dim][face][1] == 0.0  # pyrefly: ignore[unsupported-operation]
        )
        and (
            bc['v'][dim][face][0] == halo_exchange.BCType.DIRICHLET  # pyrefly: ignore[unsupported-operation]
            and bc['v'][dim][face][1] == 0.0  # pyrefly: ignore[unsupported-operation]
        )
        and (
            bc['w'][dim][face][0] == halo_exchange.BCType.DIRICHLET  # pyrefly: ignore[unsupported-operation]
            and bc['w'][dim][face][1] == 0.0  # pyrefly: ignore[unsupported-operation]
        )
    )

  def is_slip_wall(dim: int, face: int):
    """Checks if the boundary is a free-slip wall."""
    wall_normal_velocity = velocity_var(dim)

    # The velocity component normal to the wall should be 0 to have no
    # penetration.
    if (
        bc[wall_normal_velocity][dim][face][0] != halo_exchange.BCType.DIRICHLET  # pyrefly: ignore[unsupported-operation]
        or bc[wall_normal_velocity][dim][face][1] != 0.0  # pyrefly: ignore[unsupported-operation]
    ):
      return False

    for velocity in ['u', 'v', 'w']:
      if velocity == wall_normal_velocity:
        continue
      # Zero shear needs to be applied at a slip wall.
      if (
          bc[velocity][dim][face][0]  # pyrefly: ignore[unsupported-operation]
          not in (halo_exchange.BCType.NEUMANN, halo_exchange.BCType.NEUMANN_2)
          or bc[velocity][dim][face][1] != 0.0  # pyrefly: ignore[unsupported-operation]
      ):
        return False

    return True

  def is_shear_wall(dim: int, face: int):
    """Checks if the boundary is a shear wall."""
    wall_normal_velocity = velocity_var(dim)

    # The velocity component normal to the wall should be 0 to have no
    # penetration.
    if (
        bc[wall_normal_velocity][dim][face][0] != halo_exchange.BCType.DIRICHLET  # pyrefly: ignore[unsupported-operation]
        or bc[wall_normal_velocity][dim][face][1] != 0.0  # pyrefly: ignore[unsupported-operation]
    ):
      return False

    non_zero_shear = False

    for velocity in ['u', 'v', 'w']:
      if velocity == wall_normal_velocity:
        continue
      # Zero shear needs to be applied at a slip wall.
      if bc[velocity][dim][face][0] not in (  # pyrefly: ignore[unsupported-operation]
          halo_exchange.BCType.NEUMANN,
          halo_exchange.BCType.NEUMANN_2,
      ):
        return False

      if bc[velocity][dim][face][1] != 0.0:  # pyrefly: ignore[unsupported-operation]
        non_zero_shear = True

    return non_zero_shear

  def is_inflow(dim: int, face: int):
    """Checks if the boundary is an inflow."""
    mainstream = velocity_var(dim)

    for velocity in ['u', 'v', 'w']:
      bc_local = bc[velocity][dim][face]  # pyrefly: ignore[unsupported-operation]
      if velocity == mainstream:
        # The mainstream velocity in the inflow has to be specified as a
        # non-zero Dirichlet boundary condition.
        if bc_local[0] != halo_exchange.BCType.DIRICHLET or bc_local[1] == 0.0:  # pyrefly: ignore[unsupported-operation]
          return False
      else:
        # The tangential velocity components in the inflow have to be specified
        # as Dirichlet boundary condition with arbitrary values.
        if bc_local[0] != halo_exchange.BCType.DIRICHLET:  # pyrefly: ignore[unsupported-operation]
          return False

    return True

  # Note, this is currently exclusively used for deriving the BC type for
  # pressure.
  def is_outflow(dim: int, face: int):
    """Checks if the boundary is an outflow."""

    # Here we only consider the case in which the outflow is specified by an
    # all-Neumann boundary condition.
    return (
        bc['u'][dim][face][0]  # pyrefly: ignore[unsupported-operation]
        in (
            halo_exchange.BCType.NEUMANN,
            halo_exchange.BCType.NEUMANN_2,
            halo_exchange.BCType.NONREFLECTING,
        )
        and bc['v'][dim][face][0]  # pyrefly: ignore[unsupported-operation]
        in (
            halo_exchange.BCType.NEUMANN,
            halo_exchange.BCType.NEUMANN_2,
            halo_exchange.BCType.NONREFLECTING,
        )
        and bc['w'][dim][face][0]  # pyrefly: ignore[unsupported-operation]
        in (
            halo_exchange.BCType.NEUMANN,
            halo_exchange.BCType.NEUMANN_2,
            halo_exchange.BCType.NONREFLECTING,
        )
    )

  for dim in range(3):
    if periodic_dims[dim]:
      bc_type[dim] = [BoundaryType.PERIODIC, BoundaryType.PERIODIC]  # pyrefly: ignore[unsupported-operation]
      continue

    for face in range(2):
      if is_non_slip_wall(dim, face):
        bc_type[dim][face] = BoundaryType.NON_SLIP_WALL  # pyrefly: ignore[unsupported-operation]
      elif is_slip_wall(dim, face):
        bc_type[dim][face] = BoundaryType.SLIP_WALL  # pyrefly: ignore[unsupported-operation]
      elif is_shear_wall(dim, face):
        bc_type[dim][face] = BoundaryType.SHEAR_WALL  # pyrefly: ignore[unsupported-operation]
      elif is_inflow(dim, face):
        bc_type[dim][face] = BoundaryType.INFLOW  # pyrefly: ignore[unsupported-operation]
      elif is_outflow(dim, face):
        bc_type[dim][face] = BoundaryType.OUTFLOW  # pyrefly: ignore[unsupported-operation]
      else:
        bc_type[dim][face] = BoundaryType.UNKNOWN  # pyrefly: ignore[unsupported-operation]

  return bc_type  # pyrefly: ignore[bad-return]


def get_keys_for_boundary_condition(
    bc: BoundaryConditionDict, bc_type: halo_exchange.BCType
) -> list[str]:
  """Generates a list of string keys for storing boudary values for `bc_type`.

  Args:
    bc: The dictionary containing the boundary conditions.
    bc_type: The type of boundary condition to generate the keys for.

  Returns:
    A set of strings to be used as the key to `additional_states` for storing
    the corresponding boundary condition values.
  """
  keys_for_bc = []
  bc_manager = physical_variable_keys_manager.BoundaryConditionKeysHelper()
  for k, v in bc.items():
    if v is None:
      continue
    for dim in range(3):
      for face in range(2):
        if v[dim][face] is None:  # pyrefly: ignore[unsupported-operation]
          continue
        if v[dim][face][0] == bc_type:  # pyrefly: ignore[unsupported-operation]
          additional_state_key_for_bc = bc_manager.generate_bc_key(k, dim, face)
          logging.info(
              'Encountering %s BC for variable: %s, at dimension: '
              '%d and face: %d. New additional_state_key: %s is added.',
              str(bc_type),
              k,
              dim,
              face,
              additional_state_key_for_bc,
          )
          keys_for_bc.append(additional_state_key_for_bc)
  return keys_for_bc


_BC_KEY_HELPER = physical_variable_keys_manager.BoundaryConditionKeysHelper()


def is_bc_key(key: str) -> bool:
  """Checks if `key` is a standard boundary condition key (e.g., `bc_u_0_0`)."""
  return _BC_KEY_HELPER.parse_key(key) is not None


def parse_bc_key(key: str) -> tuple[str, int, int] | None:
  """Parses a boundary condition key into `(varname, dim, face)`.

  Args:
    key: The key to parse, expected to match `bc_{var}_{dim}_{face}`.

  Returns:
    A tuple of (varname, dim, face) if valid, else None.
  """
  return _BC_KEY_HELPER.parse_key(key)


def _expand_and_pad_2d_plane(
    plane: types.ScalarField,
    halo_width: int,
    pad_mode: str = 'edge',
) -> types.ScalarField:
  """Tiles a 2D plane along the normal axis and pads the in-plane dimensions."""
  tiled = jnp.tile(jnp.expand_dims(plane, 0), [halo_width + 1, 1, 1])
  return jnp.pad(
      tiled,
      pad_width=((0, 0), (halo_width, halo_width), (halo_width, halo_width)),
      mode=pad_mode,
  )


def boundary_plane_to_bc(
    plane_or_tensor: types.ScalarField,
    dim: int,
    halo_width: int,
    grid_params: grid_parametrization.GridParametrization | None = None,
    pad_mode: str = 'edge',
) -> types.ScalarField:
  """Arranges a 2D boundary plane or 3D normal profile into the 3D BC array format.

  The boundary face dimension has size `halo_width + 1` (tiled across halos
  and the boundary layer, or preserving the provided normal profile). The
  in-plane dimensions are padded with `pad_mode` for halo layers. The output is
  transposed to match `grid_params.data_axis_order`.

  Args:
    plane_or_tensor: A 2D array of shape `(n_plane_0, n_plane_1)` or a 3D array
      of shape `(halo_width + 1, n_plane_0, n_plane_1)` along the boundary.
    dim: The normal physical dimension index (0, 1, or 2) corresponding to 'x',
      'y', 'z'.
    halo_width: The width of halo layers.
    grid_params: Optional grid parametrization for data_axis_order awareness.
    pad_mode: Padding mode for in-plane halo layers (e.g. 'edge', 'constant').

  Returns:
    A 3D array formatted for boundary condition storage.
  """
  if dim not in range(3):
    raise ValueError(f'Dimension has to be one of 0, 1, and 2. Given {dim}.')
  plane_dims = [d for d in range(3) if d != dim]
  if plane_or_tensor.ndim == 2:
    boundary_tensor = _expand_and_pad_2d_plane(
        plane_or_tensor, halo_width, pad_mode=pad_mode
    )
  elif plane_or_tensor.ndim == 3:
    if plane_or_tensor.shape[0] != halo_width + 1:
      raise ValueError(
          'Expected first dimension of 3D profile to have size'
          f' {halo_width + 1}, but got {plane_or_tensor.shape[0]}.'
      )
    boundary_tensor = jnp.pad(
        plane_or_tensor,
        pad_width=((0, 0), (halo_width, halo_width), (halo_width, halo_width)),
        mode=pad_mode,
    )
  else:
    raise ValueError(
        'Expected 2D plane or 3D normal profile for boundary condition, but'
        f' got array with ndim={plane_or_tensor.ndim}.'
    )

  physical_order = [dim] + plane_dims
  if grid_params is not None:
    data_order = list(grid_params.data_axis_order)
  else:
    data_order = ['z', 'x', 'y']
  axes_names = ['x', 'y', 'z']
  perm = [
      physical_order.index(axes_names.index(data_order[j])) for j in range(3)
  ]
  return jnp.transpose(boundary_tensor, perm)


def extract_bc_face_planes(
    bc_tensor: types.ScalarField,
    dim: int,
    face: Literal[0, 1],
    halo_width: int,
    grid_params: grid_parametrization.GridParametrization,
) -> list[types.ScalarField]:
  """Extracts `halo_width` 2D face planes from a 3D boundary condition tensor.

  Args:
    bc_tensor: A 3D tensor containing boundary condition values in
      `grid_params.data_axis_order`.
    dim: Physical dimension index (0='x', 1='y', 2='z').
    face: Boundary face (0 for low, 1 for high).
    halo_width: The width of halo layers.
    grid_params: Grid parametrization object.

  Returns:
    A list of `halo_width` 2D face planes ordered from low to high coordinate,
    as expected by `halo_exchange`.
  """
  axis = ('x', 'y', 'z')[dim]
  bc_planes = []
  for i in range(halo_width):
    bc_planes.append(common_ops.get_face(bc_tensor, axis, face, i, grid_params))
  if face == 1:
    bc_planes = bc_planes[::-1]
  return bc_planes


def get_bc_partition_spec(
    key: str,
    mesh: Mesh,
    grid_params: grid_parametrization.GridParametrization | None = None,
) -> P:
  """Returns the `PartitionSpec` for a 3D boundary condition variable.

  The boundary condition array dimensions match `grid_params.data_axis_order`
  (or default `('z', 'x', 'y')` if `grid_params` is None). The normal dimension
  has size `halo_width + 1` and is unpartitioned (`None`). The other two
  dimensions (in the boundary plane) are sharded along their respective mesh
  axes.

  Args:
    key: The boundary condition key, e.g. `bc_u_0_0`.
    mesh: The JAX `Mesh` defining device layout.
    grid_params: Optional grid parametrization. If mesh axis names do not match
      physical axes ('x', 'y', 'z'), position is inferred from
      `grid_params.data_axis_order`.

  Returns:
    A `PartitionSpec` matching the rank-3 boundary condition array.

  Raises:
    ValueError: If `key` is not a valid boundary condition key.
  """
  info = parse_bc_key(key)
  if info is None:
    raise ValueError(f'{key} is not a valid boundary condition key.')
  _, dim, _ = info
  normal_axis = ('x', 'y', 'z')[dim]

  # Determine physical axis corresponding to each dimension of the 3D BC array.
  if grid_params is not None:
    data_order = list(grid_params.data_axis_order)
  elif all(a in ('x', 'y', 'z') for a in mesh.axis_names):
    data_order = list(mesh.axis_names)
  else:
    data_order = ['z', 'x', 'y']

  spec = []
  for j, axis in enumerate(data_order):
    if axis == normal_axis:
      spec.append(None)
    elif axis in mesh.axis_names:
      spec.append(axis)
    elif grid_params is not None and axis in grid_params.data_axis_order:
      data_dim = grid_params.data_axis_order.index(axis)
      spec.append(mesh.axis_names[data_dim])
    else:
      spec.append(mesh.axis_names[j])
  return P(*spec)


def get_bc_sharding(
    key: str,
    mesh: Mesh,
    grid_params: grid_parametrization.GridParametrization | None = None,
) -> NamedSharding:
  """Returns the `NamedSharding` for a boundary condition variable."""
  return NamedSharding(mesh, get_bc_partition_spec(key, mesh, grid_params))


def block_bc_field(
    key: str,
    per_replica_states: Sequence[dict[str, Any]],
    mesh: Mesh,
    grid_params: grid_parametrization.GridParametrization | None = None,
) -> jax.Array:
  """Assembles per-replica 3D BC fields into a single global array.

  The boundary face dimension has size `halo_width + 1` and is not partitioned
  across replicas. The other two dimensions (in the boundary plane) are sharded
  across replicas along their respective mesh axes and concatenated here.

  Args:
    key: The boundary condition key, e.g. `bc_u_0_0`.
    per_replica_states: Sequence of state dicts from each replica, ordered by
      replica id.
    mesh: The JAX `Mesh` defining device layout.
    grid_params: Optional grid parametrization.

  Returns:
    A global 3D array combining all per-replica shards.

  Raises:
    ValueError: If `key` is not a valid boundary condition key or not found.
  """
  info = parse_bc_key(key)
  if info is None:
    raise ValueError(f'{key} is not a valid boundary condition key.')
  _, dim, face = info
  num_replicas = len(per_replica_states)
  sample = next((s[key] for s in per_replica_states if key in s), None)
  if sample is None:
    raise ValueError(f'Key {key} not found in any replica state.')
  if num_replicas == 1:
    return jnp.asarray(sample)

  normal_axis = ('x', 'y', 'z')[dim]
  plane_dims = [d for d in range(3) if d != dim]
  axis_0 = ('x', 'y', 'z')[plane_dims[0]]
  axis_1 = ('x', 'y', 'z')[plane_dims[1]]

  # Determine physical axis ordering of the 3D BC array.
  if grid_params is not None:
    data_order = list(grid_params.data_axis_order)
  elif all(a in ('x', 'y', 'z') for a in mesh.axis_names):
    data_order = list(mesh.axis_names)
  else:
    data_order = ['z', 'x', 'y']

  array_axis_0 = data_order.index(axis_0)
  array_axis_1 = data_order.index(axis_1)

  mesh_axis_0 = (
      axis_0  # pylint: disable=g-long-ternary
      if axis_0 in mesh.axis_names
      else (
          mesh.axis_names[grid_params.data_axis_order.index(axis_0)]
          if grid_params is not None
          else mesh.axis_names[data_order.index(axis_0)]
      )
  )
  mesh_axis_1 = (
      axis_1  # pylint: disable=g-long-ternary
      if axis_1 in mesh.axis_names
      else (
          mesh.axis_names[grid_params.data_axis_order.index(axis_1)]
          if grid_params is not None
          else mesh.axis_names[data_order.index(axis_1)]
      )
  )
  mesh_normal = (
      normal_axis  # pylint: disable=g-long-ternary
      if normal_axis in mesh.axis_names
      else (
          mesh.axis_names[grid_params.data_axis_order.index(normal_axis)]
          if grid_params is not None
          else mesh.axis_names[data_order.index(normal_axis)]
      )
  )

  computation_shape = tuple(mesh.shape[name] for name in mesh.axis_names)
  c_0 = mesh.shape[mesh_axis_0]
  c_1 = mesh.shape[mesh_axis_1]

  target_normal_coord = 0 if face == 0 else (mesh.shape[mesh_normal] - 1)

  grid_bc = np.full((c_0, c_1), None, dtype=object)
  for idx in range(num_replicas):
    multi_idx = np.unravel_index(idx, computation_shape)
    replica_coords = {
        name: multi_idx[i] for i, name in enumerate(mesh.axis_names)
    }
    coord_0 = replica_coords[mesh_axis_0]
    coord_1 = replica_coords[mesh_axis_1]
    if key in per_replica_states[idx]:
      if replica_coords[mesh_normal] == target_normal_coord:
        grid_bc[coord_0, coord_1] = per_replica_states[idx][key]
      elif grid_bc[coord_0, coord_1] is None:
        grid_bc[coord_0, coord_1] = per_replica_states[idx][key]

  if any(v is None for v in grid_bc.flat):
    raise ValueError(
        f'Missing replica shards when assembling {key} on mesh {mesh.shape}.'
    )

  rows = []
  for i in range(c_0):
    row = jnp.concatenate(
        [grid_bc[i, j] for j in range(c_1)], axis=array_axis_1
    )
    rows.append(row)
  return jnp.concatenate(rows, axis=array_axis_0)


def reconstruct_bc_field(
    key: str,
    per_replica_states: Sequence[dict[str, Any]],
    mesh: Mesh,
    grid_params: grid_parametrization.GridParametrization | None = None,
) -> jax.Array:
  """Reconstructs and shards a 3D boundary condition field across replicas.

  Assembles per-replica 3D BC fields into a global array sharded with
  `NamedSharding` across the mesh.

  Args:
    key: The boundary condition key, e.g. `bc_u_0_0`.
    per_replica_states: Sequence of state dicts, one per device.
    mesh: The JAX `Mesh` defining device layout.
    grid_params: Optional grid parametrization.

  Returns:
    A globally-sharded JAX Array.
  """
  full_bc = block_bc_field(key, per_replica_states, mesh, grid_params)
  sharding_bc = get_bc_sharding(key, mesh, grid_params)
  return jax.device_put(full_bc, sharding_bc)


distribute_bc_state = reconstruct_bc_field
