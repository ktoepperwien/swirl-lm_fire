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
"""Driver for the JAX Navier-Stokes solver.

Provides utilities to parse configs, run the simulation loop with optional
checkpointing, restart from a previous checkpoint, and visualization hooks.

The main solver loop structure:

  solver_loop  (Python cycle loop, handles IO/checkpointing)
    └─ _one_cycle  (JIT-compiled, runs `num_steps` time steps)
         └─ per-step:
              1. preprocess (conditional on step_id)
              2. _update_additional_states
              3. model.step
              4. postprocess (conditional on step_id)
              5. TIME_VARNAME accumulation

Multi-device support:
  The simulation uses SPMD execution via `shard_map`. The 3D computational
  domain is partitioned across devices according to `params.cx/cy/cz`
  (matching TF's `computation_shape`). A JAX `Mesh` with axis names
  matching `data_axis_order` is constructed from `cx * cy * cz` devices.

  The step body runs inside `shard_map`, providing named-axis context for:
    - `jax.lax.ppermute` (halo exchange between neighboring devices)
    - `jax.lax.axis_index` (device boundary detection)
    - `jax.lax.psum` (global reductions in linear solvers)

  State arrays are distributed across devices using `NamedSharding`.
  Each device holds its local partition of size `(nx, ny, nz)`.

Usage example (TGV with checkpointing):
  ```python
  from swirl_lm.jax.core import driver

  config_proto = driver.load_config('tgv_3d.textpb')
  params = parameters_lib.SwirlLMParameters(config_proto)

  def tgv_init_fn(params):
    ...
    return {'rho': rho, 'u': u, 'v': v, 'w': w, 'p': p}

  states = driver.run_simulation(
      params=params,
      init_fn=tgv_init_fn,
      num_steps=1000,
      mesh=mesh,
      checkpoint_dir='/tmp/tgv_run',
      checkpoint_interval=100,
  )
  ```

Usage example (restart from a previous checkpoint):
  ```python
  # Restart from step 500 and run 500 more steps.
  states = driver.run_simulation(
      params=params,
      init_fn=tgv_init_fn,
      num_steps=500,
      mesh=mesh,
      restart_from='/tmp/tgv_run/step_000500.zarr',
      checkpoint_dir='/tmp/tgv_run_continued',
      checkpoint_interval=100,
  )
  ```
"""


from collections.abc import Callable, Mapping
import inspect
import os
import time
from typing import Any

from absl import logging
from google.protobuf import text_format
import jax
from jax.experimental import mesh_utils
from jax.experimental import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.boundary_condition import nonreflecting_boundary
from swirl_lm.jax.core import simulation as simulation_lib
from swirl_lm.jax.io import checkpoint as checkpoint_lib
from swirl_lm.jax.physics.lpt import lpt
from swirl_lm.jax.utility import grid_parametrization as gp_lib
from swirl_lm.jax.utility import stretched_grid
from swirl_lm.jax.utility import stretched_grid_util
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

TIME_VARNAME = 'simulation_time'

# Type for the user-supplied initialization function.
# It receives params and a logical_coordinate tuple (cx_i, cy_i, cz_i) for the
# device, and returns the device-local initial states dict.
# This matches TF's `init_fn(replica_id, coordinates)` pattern.
InitFn = Callable[
    [parameters_lib.SwirlLMParameters, tuple[int, int, int]],
    dict[str, ScalarField],
]


def load_config(filepath: str) -> parameters_pb2.SwirlLMParameters:
  """Loads a SwirlLMParameters proto from a textproto file.

  Args:
    filepath: Path to the textproto configuration file.

  Returns:
    The parsed SwirlLMParameters proto.
  """
  with open(filepath, 'r') as f:
    config_text = f.read()
  config = parameters_pb2.SwirlLMParameters()
  text_format.Parse(config_text, config)
  return config


def create_mesh(
    params: parameters_lib.SwirlLMParameters,
) -> Mesh:
  """Creates a JAX Mesh from the simulation parameters.

  The mesh shape is `(cx, cy, cz)` reordered to match `data_axis_order` from
  `params.grid_params`, so that mesh axis 'x' has size cx, 'y' has size cy,
  and 'z' has size cz. For example, with data_axis_order=('z', 'x', 'y'),
  the mesh shape is (cz, cx, cy).

  Args:
    params: The simulation parameters with cx, cy, cz and grid_params.

  Returns:
    A JAX Mesh suitable for shard_map execution.
  """
  gp = params.grid_params
  # Reorder (cx, cy, cz) from (x, y, z) order to data_axis_order so that
  # each mesh axis name matches its partition count.
  mesh_shape = gp.to_data_axis_order(gp.cx, gp.cy, gp.cz)
  num_devices = gp.cx * gp.cy * gp.cz
  devices = mesh_utils.create_device_mesh(
      mesh_shape, devices=jax.devices()[:num_devices]
  )
  return Mesh(devices, axis_names=gp.data_axis_order)


def _get_state_keys(
    params: parameters_lib.SwirlLMParameters,
) -> tuple[list[str], list[str], list[str]]:
  """Returns essential, additional, and helper var state keys.

  Mirrors TF `_get_state_keys`. Separates all simulation state keys into
  three categories:
    - essential_keys: prognostic flow field variables.
    - additional_keys: boundary conditions, forcing terms, TIME_VARNAME, etc.
    - helper_var_keys: Poisson solver helper vars, monitor analytics, etc.

  Args:
    params: The simulation parameters.

  Returns:
    A tuple of (essential_keys, additional_keys, helper_var_keys).
  """
  essential_keys = ['u', 'v', 'w', 'p'] + list(params.transport_scalars_names)
  if params.solver_procedure == parameters_lib.SolverProcedure.VARIABLE_DENSITY:
    essential_keys += ['rho']

  additional_keys = list(
      params.additional_state_keys if params.additional_state_keys else []
  )
  helper_var_keys = list(
      params.helper_var_keys if params.helper_var_keys else []
  )

  additional_keys.append(TIME_VARNAME)

  # Add LPT keys if configured.
  additional_keys += lpt.required_keys(params.lpt)

  # Add stretched grid keys. In JAX all fields are 3D arrays, so scale
  # factors go into helper_var_keys and 3D coordinates into additional_keys.
  coordinate_keys_3d = ('xx', 'yy', 'zz')
  use_stretched_grid_xyz = params.grid_params.to_xyz_order((
      params.use_stretched_grid[0],
      params.use_stretched_grid[1],
      params.use_stretched_grid[2],
  ))
  for dim in range(3):
    if use_stretched_grid_xyz[dim]:
      additional_keys.append(coordinate_keys_3d[dim])
      helper_var_keys.append(stretched_grid_util.h_key(dim))
      helper_var_keys.append(stretched_grid_util.h_face_key(dim))

  # Check for duplicates.
  all_keys = essential_keys + additional_keys + helper_var_keys
  if len(set(essential_keys)) + len(set(additional_keys)) + len(
      set(helper_var_keys)
  ) != len(set(all_keys)):
    raise ValueError(
        'Duplicated keys detected between the three types of states: '
        f'essential states: {essential_keys}, additional states: '
        f'{additional_keys}, and helper vars: {helper_var_keys}'
    )

  return essential_keys, additional_keys, helper_var_keys


def _logical_coordinates(
    params: parameters_lib.SwirlLMParameters,
) -> list[tuple[int, int, int]]:
  """Generates logical coordinates for all replicas.

  Mirrors TF's `tpu_util.grid_coordinates(computation_shape)`. Each replica
  gets a `(cx_i, cy_i, cz_i)` coordinate identifying its position in the
  3D computational grid.

  The iteration order must match `_distribute_states`, which uses
  `np.unravel_index` over `computation_shape` (data_axis_order). We iterate
  in C-order over data_axis_order dimensions and convert each position
  back to physical `(ix, iy, iz)` coordinates.

  Args:
    params: The simulation parameters.

  Returns:
    A list of `(cx_i, cy_i, cz_i)` tuples, one per replica, ordered by
    replica id (C-order over data_axis_order dimensions).
  """
  gp = params.grid_params
  # computation_shape in data_axis_order, matching _distribute_states.
  c_data = gp.to_data_axis_order(gp.cx, gp.cy, gp.cz)
  coords = []
  for i0 in range(c_data[0]):
    for i1 in range(c_data[1]):
      for i2 in range(c_data[2]):
        # Convert from data_axis_order position to physical (ix, iy, iz).
        xyz = gp.to_xyz_order((i0, i1, i2))
        coords.append(xyz)
  return coords


def _init_fn(
    params: parameters_lib.SwirlLMParameters,
    logical_coordinates: tuple[int, int, int],
    customized_init_fn: InitFn | None = None,
) -> dict[str, ScalarField]:
  """Generates the initial state for a single device.

  Mirrors TF `_init_fn` which produces a per-replica init function.
  Each device calls this with its own `logical_coordinates` to get
  its device-local state partition.

  Args:
    params: The simulation parameters.
    logical_coordinates: `(cx_i, cy_i, cz_i)` position of this device in the
      computational grid.
    customized_init_fn: Optional user-provided init function. It receives
      `(params, logical_coordinates)` and returns a dict of device-local state
      arrays.

  Returns:
    A dict of initial states for this device.
  """
  states: dict[str, ScalarField] = {}

  # Initialize stretched grid variables (scale factors and 3D coordinates).
  if any(params.use_stretched_grid):
    stretched_vars = stretched_grid.local_stretched_grid_vars_from_global_xyz(
        params, logical_coordinates=logical_coordinates
    )
    states.update(stretched_vars)

  # Initialize TIME_VARNAME.
  states[TIME_VARNAME] = jnp.float64(0.0)

  # Add LPT variables if configured.
  if params.lpt is not None:
    states.update(lpt.init_fn(params))

  # Apply user-defined init_fn last to allow overrides.
  if customized_init_fn is not None:
    states.update(customized_init_fn(params, logical_coordinates))

  # Add nonreflecting BC states.
  states.update(nonreflecting_boundary.nonreflecting_bc_state_init_fn(params))

  return states


def _distribute_states(
    per_replica_states: list[dict[str, ScalarField]],
    mesh: Mesh,
    grid_params: gp_lib.GridParametrization | None = None,
    params: parameters_lib.SwirlLMParameters | None = None,
) -> dict[str, jax.Array]:
  """Assembles per-replica states into globally-sharded JAX arrays.

  Each device produces its local state partition via `_init_fn`. This
  function concatenates them into global arrays with `NamedSharding`
  so that each device's shard is placed on the correct device.

  Scalar values (0D arrays like TIME_VARNAME) are replicated across all
  devices rather than sharded.

  Args:
    per_replica_states: List of state dicts, one per device, ordered by replica
      id.
    mesh: The JAX mesh defining device layout.
    grid_params: Optional grid parametrization.
    params: Optional SwirlLMParameters containing custom partition specs and
      distributors.

  Returns:
    A single state dict of globally-sharded JAX arrays.
  """
  all_keys = list(dict.fromkeys([k for s in per_replica_states for k in s]))  # pylint: disable=g-complex-comprehension
  sharding_3d = NamedSharding(mesh, P(*mesh.axis_names))
  sharding_replicated = NamedSharding(mesh, P())
  computation_shape = tuple(mesh.shape[name] for name in mesh.axis_names)
  num_replicas = len(per_replica_states)

  global_state: dict[str, jax.Array] = {}

  for key in all_keys:
    sample = next((s[key] for s in per_replica_states if key in s), None)
    if sample is None:
      continue

    if boundary_condition_utils.is_bc_key(key):
      global_state[key] = boundary_condition_utils.distribute_bc_state(
          key, per_replica_states, mesh, grid_params
      )
    elif params is not None and key in params.additional_state_distributors:
      global_state[key] = params.additional_state_distributors[key](
          key, per_replica_states, mesh, grid_params
      )
    elif params is not None and key in params.additional_state_partition_specs:
      spec = params.additional_state_partition_specs[key]
      sharding = NamedSharding(mesh, spec)
      global_state[key] = jax.device_put(jnp.asarray(sample), sharding)
    elif np.ndim(sample) < 3:
      # Scalar or low-rank (e.g. TIME_VARNAME, boundary face arrays):
      # replicate across all devices. All replicas should have the same value.
      global_state[key] = jax.device_put(
          jnp.asarray(sample), sharding_replicated
      )
    elif num_replicas == 1:
      # Single-device case: no concatenation needed.
      global_state[key] = jax.device_put(jnp.asarray(sample), sharding_3d)
    else:
      # Multi-device 3D field: concatenate per-device shards into global array.
      # per_replica_states are ordered in C-order over (cx, cy, cz).
      # Build a (cx, cy, cz) grid of local arrays, then concatenate along
      # each spatial axis to form the global array.
      grid = np.empty(computation_shape, dtype=object)
      for idx in range(num_replicas):
        multi_idx = np.unravel_index(idx, computation_shape)
        grid[multi_idx] = per_replica_states[idx][key]

      global_state[key] = jax.device_put(jnp.block(grid.tolist()), sharding_3d)

  return global_state


def _shard_global_additional_states(
    additional_states: Mapping[str, Any],
    mesh: Mesh,
    params: parameters_lib.SwirlLMParameters,
) -> dict[str, jax.Array]:
  """Places pre-existing, globally assembled caller arrays onto devices.

  This function places pre-existing, globally assembled caller arrays onto
  devices, whereas `_distribute_states` stitches together per-core subdomain
  tiles.

  Args:
    additional_states: User-provided dictionary of additional state variables.
    mesh: The device mesh.
    params: Simulation parameters.

  Returns:
    A dictionary of sharded device arrays.
  """
  sharding_3d = NamedSharding(mesh, P(*mesh.axis_names))
  sharding_replicated = NamedSharding(mesh, P())
  sharded_states: dict[str, jax.Array] = {}
  for key, val in additional_states.items():
    if boundary_condition_utils.is_bc_key(key):
      spec = boundary_condition_utils.get_bc_partition_spec(
          key, mesh, params.grid_params
      )
      sharded_states[key] = jax.device_put(
          jnp.asarray(val), NamedSharding(mesh, spec)
      )
    elif key in params.additional_state_partition_specs:
      spec = params.additional_state_partition_specs[key]
      sharded_states[key] = jax.device_put(
          jnp.asarray(val), NamedSharding(mesh, spec)
      )
    elif np.ndim(val) == 0:
      sharded_states[key] = jax.device_put(
          jnp.asarray(val), sharding_replicated
      )
    elif np.ndim(val) == 3:
      sharded_states[key] = jax.device_put(jnp.asarray(val), sharding_3d)
    else:
      sharded_states[key] = jax.device_put(
          jnp.asarray(val), sharding_replicated
      )
  return sharded_states


def _stateless_update_if_present(
    mapping: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
  """Returns a copy of `mapping` with only existing keys updated."""
  result = mapping.copy()
  result.update({key: val for key, val in updates.items() if key in result})
  return result


def _update_additional_states(
    essential_states: ScalarFieldMap,
    additional_states: dict[str, ScalarField],
    step_id: jax.Array,
    params: parameters_lib.SwirlLMParameters,
    mesh: Mesh,
) -> dict[str, ScalarField]:
  """Updates additional_states at each time step.

  Mirrors TF `_update_additional_states`. Performs:
    1. Clear `src_*` source terms from the previous step.
    2. Update nonreflecting BC states.
    3. Update LPT states.
    4. Call user-defined `additional_states_update_fn`.

  Args:
    essential_states: The essential (prognostic) states.
    additional_states: The additional states to update.
    step_id: The current step index.
    params: The simulation parameters.
    mesh: JAX mesh for device topology.

  Returns:
    Updated additional_states dict.
  """
  del mesh

  updated = dict(additional_states)

  # 1. Clear source terms from the previous step.
  for varname in updated:
    if varname.startswith('src_'):
      updated[varname] = jnp.zeros_like(updated[varname])

  # 2. Update nonreflecting BC states.
  updated.update(
      nonreflecting_boundary.nonreflecting_bc_state_update_fn(
          params, essential_states, additional_states, step_id  # pyrefly: ignore[bad-argument-type]
      )
  )

  # 3. Update LPT states.
  # NOTE: LPT.step() requires (replica_id, replicas, states,
  # additional_states, step_id) which need multi-device context via
  # shard_map. When running inside shard_map, replica_id and replicas
  # are available from jax.lax.axis_index. This integration is deferred
  # until LPT is fully ported.
  # TODO(wqing): Integrate LPT step with shard_map context.

  # 4. Call user-defined additional_states_update_fn.
  if params.additional_states_update_fn is not None:
    fn = params.additional_states_update_fn
    try:
      sig = inspect.signature(fn)
      accepts_params = 'params' in sig.parameters or any(
          p.kind == inspect.Parameter.VAR_KEYWORD
          for p in sig.parameters.values()
      )
    except (ValueError, TypeError):
      accepts_params = False

    if accepts_params:
      updated = dict(
          fn(
              states=essential_states,
              additional_states=additional_states,
              step_id=step_id,
              params=params,
          )
      )
    else:
      updated = dict(
          fn(
              states=essential_states,
              additional_states=additional_states,
              step_id=step_id,
          )
      )

  return updated


def _process_at_step_id(
    process_fn: Callable[..., ScalarFieldMap],
    essential_states: dict[str, ScalarField],
    additional_states: dict[str, ScalarField],
    step_id: jax.Array,
    process_step_id: int,
    is_periodic: bool,
) -> tuple[dict[str, ScalarField], dict[str, ScalarField]]:
  """Executes `process_fn` conditionally depending on `step_id`.

  Mirrors TF `_process_at_step_id`.

  Args:
    process_fn: Function accepting `states` and `additional_states` kwargs,
      returning updated states.
    essential_states: The essential (prognostic) states.
    additional_states: The additional states.
    step_id: The current step id.
    process_step_id: Step id at which to trigger execution.
    is_periodic: If True, trigger whenever step_id % process_step_id == 0.

  Returns:
    Updated (essential_states, additional_states).
  """
  should_process = (
      (step_id % process_step_id == 0)
      if is_periodic
      else step_id == process_step_id
  )

  def _do_process(ess, add):
    updated_states = dict(process_fn(states=ess, additional_states=add))
    return (
        _stateless_update_if_present(ess, updated_states),
        _stateless_update_if_present(add, updated_states),
    )

  def _skip_process(ess, add):
    return ess, add

  essential_states, additional_states = jax.lax.cond(
      should_process,
      _do_process,
      _skip_process,
      essential_states,
      additional_states,
  )

  return essential_states, additional_states


def _save_checkpoint(
    states: dict[str, ScalarField],
    grid_params: gp_lib.GridParametrization,
    step: int,
    checkpoint_dir: str,
) -> None:
  """Saves a checkpoint to the given directory.

  Args:
    states: Flow field variables.
    grid_params: Grid parametrization for coordinate metadata.
    step: Current simulation step number.
    checkpoint_dir: Directory for zarr checkpoints.
  """
  ckpt_path = os.path.join(checkpoint_dir, f'step_{step:06d}.zarr')
  checkpoint_lib.save(
      states,
      grid_params,
      step=step,
      path=ckpt_path,
      sim_time=step * grid_params.dt,
  )
  logging.info('Checkpoint saved: %s', ckpt_path)


def run_simulation(
    params: parameters_lib.SwirlLMParameters,
    init_fn: InitFn,
    num_steps: int,
    mesh: Mesh | None = None,
    additional_states: ScalarFieldMap | None = None,
    restart_from: str | None = None,
    num_steps_per_cycle: int = 1,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 0,
) -> dict[str, ScalarField]:
  """Runs the simulation for the given number of steps.

  Mirrors TF `solver` + `solver_loop`. Structure:
    - Create JAX mesh from params.cx/cy/cz (if not provided).
    - Initialize per-device states via `_init_fn` with logical_coordinates.
    - Distribute states across devices using NamedSharding.
    - Run cycles, each calling `_one_cycle` (JIT + shard_map compiled).
    - IO/checkpointing happens between cycles on host.

  The step body runs inside `shard_map` to provide named-axis context for
  collective ops (ppermute, axis_index, psum). This mirrors TF's
  `strategy.run(step_fn, args=(init_state,))`.

  Args:
    params: The simulation parameters.
    init_fn: A callable `(params, logical_coordinates) -> dict[str,
      ScalarField]` that returns per-device initial states. The
      `logical_coordinates` is a `(cx_i, cy_i, cz_i)` tuple identifying the
      device's position in the computational grid, matching TF's
      `init_fn(replica_id, coordinates)` pattern.
    num_steps: Total number of time steps to run.
    mesh: Optional JAX mesh for device topology. If None, it is created
      automatically from `params.cx/cy/cz` via `create_mesh(params)`.
    additional_states: Optional helper variables (boundary conditions, stretched
      grid scale factors, etc.). Empty dict if None.
    restart_from: Path to a zarr checkpoint to restart from. Variables present
      in the checkpoint override those from `init_fn`.
    num_steps_per_cycle: Number of time steps per cycle. Higher values reduce
      Python loop overhead but delay IO. Default is 1.
    checkpoint_dir: Directory for zarr checkpoints. If None, no checkpointing.
    checkpoint_interval: Save a checkpoint every N cycles. 0 = no periodic
      saves.

  Returns:
    The final states dict (globally-sharded arrays).

  Raises:
    RuntimeError: If the simulation diverges (non-finite values in velocity).
  """
  # Create mesh from params if not provided.
  if mesh is None:
    mesh = create_mesh(params)

  # Initialize per-device states.
  coords = _logical_coordinates(params)
  per_replica_states = []
  for lc in coords:
    device_state = _init_fn(params, lc, customized_init_fn=init_fn)
    per_replica_states.append(device_state)

  # Distribute across devices.
  state = _distribute_states(
      per_replica_states, mesh, params.grid_params, params=params
  )

  # Merge in user-provided additional_states.
  if additional_states is not None:
    state.update(
        _shard_global_additional_states(additional_states, mesh, params)
    )

  # Override with checkpoint data if restarting.
  if restart_from is not None:
    ckpt_states, metadata = checkpoint_lib.load(restart_from)
    overridden = []
    for key, value in ckpt_states.items():
      if key in state:
        state[key] = value
        overridden.append(key)
      else:
        logging.warning(
            'Checkpoint variable %r not in init states; skipping.', key
        )
    logging.info(
        'Restarted from checkpoint %s (step %s). Overrode variables: %s',
        restart_from,
        metadata.get('step', '?'),
        overridden,
    )

  # Create simulation model.
  sim = simulation_lib.Simulation(params)
  gp = params.grid_params

  # Get key categorization.
  essential_keys, additional_keys, helper_var_keys = _get_state_keys(params)

  # Log key overlap info.
  state_keys = set(state.keys())
  declared_keys = (
      set(essential_keys) | set(additional_keys) | set(helper_var_keys)
  )
  logging.info(
      'Keys in state but not in params: %s',
      sorted(state_keys - declared_keys),
  )
  logging.info(
      'Keys in params but not in state: %s',
      sorted(declared_keys - state_keys),
  )

  # Build the JIT-compiled cycle function with shard_map.
  # The cycle function runs `num_steps_per_cycle` time steps. Inside each step:
  #   1. Split state -> essential + additional
  #   2. Preprocess (conditional on step_id)
  #   3. _update_additional_states
  #   4. sim.step
  #   5. Postprocess (conditional on step_id)
  #   6. TIME_VARNAME += dt
  #   7. Merge back into flat state
  #
  # This matches the TF `_one_cycle` structure. The step body runs inside
  # shard_map so that collective ops (ppermute, axis_index, psum) have
  # named-axis context, matching TF's `strategy.run(step_fn, ...)`.

  # Build shard_map specs: rank-3 fields get P(*axis_names).
  # Boundary condition and helper states get their model-specific PartitionSpec.
  # All other ranks (scalars, 1D arrays) are replicated.
  all_state_keys = sorted(state.keys())
  spec_3d = P(*mesh.axis_names)
  spec_replicated = P()
  in_specs = {}
  out_specs = {}
  for key in all_state_keys:
    if boundary_condition_utils.is_bc_key(key):
      spec = boundary_condition_utils.get_bc_partition_spec(
          key, mesh, params.grid_params
      )
      in_specs[key] = spec
      out_specs[key] = spec
    elif key in params.additional_state_partition_specs:
      spec = params.additional_state_partition_specs[key]
      in_specs[key] = spec
      out_specs[key] = spec
    elif np.ndim(state[key]) == 3:
      in_specs[key] = spec_3d
      out_specs[key] = spec_3d
    else:
      in_specs[key] = spec_replicated
      out_specs[key] = spec_replicated

  def _shard_map_cycle_body(
      state: dict[str, ScalarField],
      init_step_id: jax.Array,
  ) -> dict[str, ScalarField]:
    """Step body that runs inside shard_map (per-device)."""

    def step_body(
        carry: tuple[dict[str, ScalarField], jax.Array],
        _: None,
    ) -> tuple[tuple[dict[str, ScalarField], jax.Array], None]:
      state_carry, step_id = carry

      # Split state into essential and additional.
      ess = {k: state_carry[k] for k in essential_keys if k in state_carry}
      add = {k: v for k, v in state_carry.items() if k not in ess}

      # 1. Preprocess.
      if params.apply_preprocess and params.preprocessing_states_update_fn:
        ess, add = _process_at_step_id(
            process_fn=params.preprocessing_states_update_fn,
            essential_states=ess,
            additional_states=add,
            step_id=step_id,
            process_step_id=params.preprocess_step_id,
            is_periodic=params.preprocess_periodic,
        )

      # 2. Update additional states.
      add = _update_additional_states(ess, add, step_id, params, mesh)

      # 3. Run one simulation step.
      updated_state = sim.step(ess, add, mesh)

      # 4. Postprocess.
      if params.apply_postprocess and params.postprocessing_states_update_fn:
        # Split updated_state back into essential/additional.
        add = _stateless_update_if_present(add, updated_state)
        ess = _stateless_update_if_present(ess, updated_state)

        ess, add = _process_at_step_id(
            process_fn=params.postprocessing_states_update_fn,
            essential_states=ess,
            additional_states=add,
            step_id=step_id,
            process_step_id=params.postprocess_step_id,
            is_periodic=params.postprocess_periodic,
        )

        # Merge back.
        updated_state = _stateless_update_if_present(updated_state, ess)

      # 5. Merge additional states updates into updated_state.
      updated_state.update(add)

      # 6. Accumulate simulation time.
      updated_state[TIME_VARNAME] = state_carry[TIME_VARNAME] + jnp.float64(
          params.dt
      )

      # 7. Merge updates back into state (pass-through keys like helper vars).
      new_state = _stateless_update_if_present(state_carry, updated_state)

      return (new_state, step_id + 1), None

    init_carry = (state, init_step_id)
    (final_state, _), _ = jax.lax.scan(
        step_body, init_carry, None, length=num_steps_per_cycle
    )
    return final_state

  # Build the JIT-compiled shard_map function.
  # init_step_id is a scalar, replicated across all devices.
  _jit_one_cycle = jax.jit(
      shard_map.shard_map(
          _shard_map_cycle_body,
          mesh=mesh,
          in_specs=(in_specs, spec_replicated),
          out_specs=out_specs,
          check_rep=False,
      )
  )

  # Compute cycle counts. TF uses exact `num_steps * num_cycles` — no remainder.
  num_cycles = num_steps // max(num_steps_per_cycle, 1)

  logging.info(
      'Simulation: %d total steps, %d steps/cycle, %d cycles, mesh=%s.',
      num_steps,
      num_steps_per_cycle,
      num_cycles,
      dict(mesh.shape),
  )

  # Main simulation loop (Python cycle loop).
  steps_completed = 0

  for cycle_idx in range(num_cycles):
    init_step_id = jnp.int32(steps_completed)

    # Time the compute portion of each cycle.
    compute_start = time.time()
    state = _jit_one_cycle(state, init_step_id)
    # Block until computation completes (for accurate timing).
    jax.block_until_ready(state)
    compute_time = time.time() - compute_start

    steps_completed += num_steps_per_cycle

    # Check for divergence (NaN/Inf in velocity fields).
    for vel_key in ('u', 'v', 'w'):
      if vel_key in state and not jnp.all(jnp.isfinite(state[vel_key])):
        if checkpoint_dir:
          _save_checkpoint(state, gp, steps_completed, checkpoint_dir)
          logging.error(
              'Non-finite values detected in %s at step %d. Checkpoint saved.',
              vel_key,
              steps_completed,
          )
        raise RuntimeError(
            f'Simulation diverged: non-finite values in {vel_key} at step'
            f' {steps_completed}.'
        )

    # Per-cycle logging with timing info.
    io_start = time.time()
    if num_steps_per_cycle > 0:
      gs = params.grid_spacings
      cfl_field = (
          jnp.abs(state['u']) / gs[0]
          + jnp.abs(state['v']) / gs[1]
          + jnp.abs(state['w']) / gs[2]
      )
      cfl = float(params.dt * jnp.max(cfl_field))
      logging.info(
          'Cycle %d: step %d / %d  CFL=%.4f  compute_time=%.2fs',
          cycle_idx,
          steps_completed,
          num_steps,
          cfl,
          compute_time,
      )

    # Checkpointing.
    if (
        checkpoint_dir
        and checkpoint_interval > 0
        and (cycle_idx + 1) % checkpoint_interval == 0
    ):
      _save_checkpoint(state, gp, steps_completed, checkpoint_dir)

    io_time = time.time() - io_start
    logging.info(
        'Cycle %d: io_time=%.2fs',
        cycle_idx,
        io_time,
    )

  # Save final checkpoint.
  if checkpoint_dir:
    _save_checkpoint(state, gp, num_steps, checkpoint_dir)

  logging.info('Simulation completed after %d steps.', num_steps)
  return state
