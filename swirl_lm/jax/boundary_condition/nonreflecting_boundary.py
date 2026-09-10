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
"""A library to handle nonreflecting boundary condition (JAX).

This is the JAX port of `swirl_lm.boundary_condition.nonreflecting_boundary`.

Nonreflecting BCs use a forward-Euler upwinding scheme to advect the boundary
values outward, preventing spurious reflections at outflow boundaries.

The discrete update on the "high" face is:

    phi_j^{n+1} = (1 - CFL) * phi_j^n + CFL * phi_{j-1}^n

where CFL = |U*| * dt / dx, and U* is the phase velocity at the boundary.

Three modes for computing U*:
  - NONREFLECTING_LOCAL_MAX: per-point velocity at the boundary face.
  - NONREFLECTING_GLOBAL_MEAN: global mean of the velocity at the boundary.
  - NONREFLECTING_GLOBAL_MAX: global min (face=0) or max (face=1) of velocity.
"""


from absl import logging
import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.base import physical_variable_keys_manager
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.equations import common
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# Re-export the BCParams type alias.
BCParams = physical_variable_keys_manager.BoundaryConditionKeysHelper

# Nonreflecting BC mode constants from the proto.
_NONREFLECTING_LOCAL_MAX = 0
_NONREFLECTING_GLOBAL_MEAN = 1
_NONREFLECTING_GLOBAL_MAX = 2


def _get_face_slice(
    field: ScalarField,
    dim: int,
    face: int,
    index: int,
) -> ScalarField:
  """Extracts a 1-cell-thick slice from `field` at the given face.

  Args:
    field: A 3D array of shape (nz, nx, ny).
    dim: The dimension (0=z, 1=x, 2=y in data-axis order).
    face: 0 for the lower face, 1 for the upper face.
    index: Number of cells from the boundary. 0 = first interior cell.

  Returns:
    A 3D array with size 1 along `dim`.
  """
  if face == 0:
    idx = index
  else:
    idx = field.shape[dim] - 1 - index
  slices = [slice(None)] * 3
  slices[dim] = slice(idx, idx + 1)
  return field[tuple(slices)]


def _strip_halos(field: ScalarField, halos: list[int]) -> ScalarField:
  """Strips halo cells from each dimension.

  Args:
    field: A 3D array.
    halos: Number of halo cells to strip from each dimension.

  Returns:
    The interior portion of the field.
  """
  slices = tuple(
      slice(h, s - h) if h > 0 else slice(None)
      for h, s in zip(halos, field.shape)
  )
  return field[slices]


def nonreflecting_bc_state_init_fn(
    params: parameters_lib.SwirlLMParameters,
) -> ScalarFieldMap:
  """Initializes states used in nonreflecting boundary calculations.

  For each variable/dim/face that has a NONREFLECTING BC, creates a zero
  array of shape matching the boundary condition format.

  Args:
    params: Simulation parameters.

  Returns:
    A dict mapping BC keys to zero-initialized boundary value arrays.
  """
  bc_map = {}
  gp = params.grid_params
  bc_manager = physical_variable_keys_manager.BoundaryConditionKeysHelper()
  for k in boundary_condition_utils.get_keys_for_boundary_condition(
      params.bc, halo_exchange.BCType.NONREFLECTING
  ):
    bc_info = bc_manager._parse_key(k)  # pylint: disable=protected-access
    dim = bc_info[1]
    plane_dims = [d for d in range(3) if d != dim]
    plane_core_n = [(gp.core_nx, gp.core_ny, gp.core_nz)[d] for d in plane_dims]
    zeros_2d = jnp.zeros(plane_core_n, dtype=jnp.float32)
    bc_map[k] = boundary_condition_utils.boundary_plane_to_bc(
        zeros_2d, dim, params.halo_width, gp, pad_mode='edge'
    )
  return bc_map


def nonreflecting_bc_state_update_fn(
    params: parameters_lib.SwirlLMParameters,
    states: ScalarFieldMap,
    additional_states: ScalarFieldMap,
    step_id: int,
) -> ScalarFieldMap:
  """Updates states used in nonreflecting boundary calculations.

  Uses forward-Euler upwinding to advect boundary values out of the domain.

  Args:
    params: Simulation parameters.
    states: Flow field variables. Must contain the prognostic variable and the
      velocity field for the corresponding dimension.
    additional_states: Must contain the BC state arrays (keyed by BC keys).
    step_id: The current simulation step index.

  Returns:
    A dict mapping BC keys to updated boundary value arrays.

  Raises:
    NotImplementedError: If stretched grids are used.
    ValueError: If an unsupported nonreflecting BC mode is encountered.
  """
  updated_additional_states: dict[str, ScalarField] = {}
  spacings = params.grid_spacings
  halo_width = params.halo_width
  bc_manager = physical_variable_keys_manager.BoundaryConditionKeysHelper()

  for k in boundary_condition_utils.get_keys_for_boundary_condition(
      params.bc, halo_exchange.BCType.NONREFLECTING
  ):
    if any(params.use_stretched_grid):
      raise NotImplementedError(
          'Stretched grid is not yet supported for nonreflecting boundary'
          ' condition.'
      )

    varname, dim, face = bc_manager._parse_key(k)  # pylint: disable=protected-access
    _, u_threshold = params.bc[varname][dim][face]  # pyrefly: ignore[unsupported-operation]
    bc_params_entry = params.bc_params[varname][dim][face]  # pyrefly: ignore[unsupported-operation]

    mode = (
        bc_params_entry.nonreflecting_bc_mode
        if bc_params_entry is not None
        else _NONREFLECTING_LOCAL_MAX
    )
    buffer_init_step = (
        bc_params_entry.buffer_init_step if bc_params_entry is not None else 0
    )

    logging.info(
        'Variable: %s, dimension: %d, face: %d is specified with '
        'nonreflecting bc, velocity threshold: %f, mode: %d',
        varname,
        dim,
        face,
        u_threshold,
        mode,
    )

    phi = states[varname]
    axis = ('x', 'y', 'z')[dim]

    # Get velocity at the boundary face (first interior cell).
    velocity_keys = [common.KEY_U, common.KEY_V, common.KEY_W]
    u_face_2d = common_ops.get_face(
        states[velocity_keys[dim]], axis, face, halo_width, params.grid_params
    )
    u_face = jnp.expand_dims(u_face_2d, axis=dim)

    # Compute phase velocity.
    phase_u = _phase_velocity(
        u_face, u_threshold, dim, face, mode, halo_width, spacings
    )

    # CFL number.
    cfl = jnp.abs(jnp.squeeze(phase_u, axis=dim)) * params.dt / spacings[dim]

    # Inner boundary value of phi.
    phi_inner = common_ops.get_face(
        phi, axis, face, halo_width, params.grid_params
    )

    # Previous boundary state (from additional_states).
    prev_bc = common_ops.get_face(
        additional_states[k], axis, face, 0, params.grid_params
    )

    # Forward-Euler upwind update.
    if step_id == buffer_init_step:
      # Initialize: copy the inner boundary value.
      updated = phi_inner
    else:
      updated = cfl * phi_inner + (1.0 - cfl) * prev_bc

    updated_core = (
        updated[halo_width:-halo_width, halo_width:-halo_width]
        if halo_width > 0
        else updated
    )
    updated_additional_states[k] = (
        boundary_condition_utils.boundary_plane_to_bc(
            updated_core, dim, halo_width, params.grid_params, pad_mode='edge'
        )
    )

  return updated_additional_states


def _phase_velocity(
    u_face: ScalarField,
    u_threshold: float,
    dim: int,
    face: int,
    mode: int,
    halo_width: int,
    spacings: tuple[float, ...],
) -> ScalarField:
  """Calculates the convection phase velocity for the boundary.

  Args:
    u_face: Velocity at the boundary face. Shape with size 1 along dim.
    u_threshold: The velocity threshold value from the BC config.
    dim: Dimension of the boundary.
    face: 0 for low side, 1 for high side.
    mode: Nonreflecting BC mode (LOCAL_MAX, GLOBAL_MEAN, or GLOBAL_MAX).
    halo_width: Number of halo cells.
    spacings: Grid spacings per dimension.

  Returns:
    The phase velocity field, same shape as u_face.

  Raises:
    ValueError: If mode is unsupported.
  """
  del spacings  # Unused.

  if mode == _NONREFLECTING_LOCAL_MAX:
    if face == 0:
      return jnp.minimum(u_face - u_threshold, 0.0)
    else:
      return jnp.maximum(u_face + u_threshold, 0.0)

  # For global modes, strip halos before reducing.
  halos_to_strip = [halo_width] * 3
  halos_to_strip[dim] = 0
  u_inner = _strip_halos(u_face, halos_to_strip)

  sign = -1.0 if face == 0 else 1.0

  if mode == _NONREFLECTING_GLOBAL_MEAN:
    reduced = jnp.mean(u_inner)
  elif mode == _NONREFLECTING_GLOBAL_MAX:
    reduced = jnp.min(u_inner) if face == 0 else jnp.max(u_inner)
  else:
    raise ValueError(
        f'Unsupported nonreflecting BC mode: {mode} for dim={dim}, face={face}'
    )

  # Prevent backflow.
  reduced = (
      jnp.minimum(reduced, 0.0) if face == 0 else jnp.maximum(reduced, 0.0)
  )

  return jnp.ones_like(u_face) * (reduced + sign * u_threshold)
