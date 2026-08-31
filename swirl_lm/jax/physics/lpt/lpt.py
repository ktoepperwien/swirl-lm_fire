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
"""Lagrangian particle tracking models (JAX port).

JAX port of `swirl_lm.physics.lpt.lpt`.

Particles are modeled as points in space with one-way coupling to the
surrounding fluid. Governing equations are:

  d(x_p) / dt = v_p,
  d(v_p) / dt = -c_d / tau_p * (v_p - v_f),
  d(m_p) / dt = -omega,

where `x_p` is particle location, `v_p` is particle velocity, `m_p` is
particle mass.
"""


import abc

import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.numerics import time_integration
from swirl_lm.jax.physics.lpt import injector
from swirl_lm.jax.physics.lpt import lpt_types
from swirl_lm.jax.utility import stretched_grid_util
from swirl_lm.jax.utility import types
from swirl_lm.numerics import numerics_pb2
from swirl_lm.physics import constants
from swirl_lm.physics.lpt import lpt_pb2

ScalarFieldMap = types.ScalarFieldMap


class LPT(abc.ABC):
  """Methods to manage particle fields, including positions, velocities, etc.

  The particles states are assumed to be stored in two 2D arrays:

    `lpt_field_ints`: (n, 2) for n total particle spaces.
      active_particles: 1=active, 0=inactive.
      particle_ids: Global ID of the particles.

    `lpt_field_floats`: (n, 7) for n total spaces.
      particle_locations: x0, x1, x2.
      particle_velocities: v0, v1, v2.
      particle_masses: Mass of the particles.

  Attributes:
    dt: Time step size [s].
    grid_spacings_zxy: Uniform grid spacings in the z, x, y directions.
    use_stretched_grid_zxy: Whether to use stretched grid in z, x, y.
    global_min_pt: The global minimum point in order z, x, y.
    core_spacings: Each core's partial domain size in z, x, y.
    num_replicas: The number of replicas globally.
    c_d: Drag coefficient [-].
    tau_p: Relaxation time [s].
    mass_threshold: Mass below which a particle is terminated [kg].
    n_max: Maximum number of particles each replica can have.
    gravity_direction: The gravity direction in z, x, y.
    params: The simulation parameters.
    injectors: The LptInjector types for particle injection.
  """

  def __init__(self, params: parameters_lib.SwirlLMParameters):
    if params.lpt is None:
      raise ValueError('LPT init called but lpt params are None.')

    gp = params.grid_params
    self.dt = gp.dt
    self.grid_spacings_zxy = jnp.array(
        (gp.grid_spacings[2], gp.grid_spacings[0], gp.grid_spacings[1]),
        dtype=lpt_types.LPT_FLOAT,
    )
    self.use_stretched_grid_zxy = np.array((
        gp.use_stretched_grid[2],
        gp.use_stretched_grid[0],
        gp.use_stretched_grid[1],
    ))

    self.global_min_pt = jnp.array(
        (
            0.0 if gp.use_stretched_grid[2] else float(gp.z[0]),
            0.0 if gp.use_stretched_grid[0] else float(gp.x[0]),
            0.0 if gp.use_stretched_grid[1] else float(gp.y[0]),
        ),
        dtype=lpt_types.LPT_FLOAT,
    )
    self.core_spacings = jnp.array(
        (
            (len(gp.z) - 1.0) / gp.cz
            if gp.use_stretched_grid[2]
            else gp.lz / gp.cz,
            (len(gp.x) - 1.0) / gp.cx
            if gp.use_stretched_grid[0]
            else gp.lx / gp.cx,
            (len(gp.y) - 1.0) / gp.cy
            if gp.use_stretched_grid[1]
            else gp.ly / gp.cy,
        ),
        dtype=lpt_types.LPT_FLOAT,
    )
    self.num_replicas = gp.cx * gp.cy * gp.cz

    self.c_d = params.lpt.c_d
    self.tau_p = params.lpt.tau_p
    self.mass_threshold = params.lpt.mass_threshold
    self.n_max = params.lpt.n_max

    # Compute gravity direction from proto config.
    config = params.swirl_lm_parameters_proto
    if config.HasField('gravity_direction'):
      gd = [
          config.gravity_direction.dim_0,
          config.gravity_direction.dim_1,
          config.gravity_direction.dim_2,
      ]
      g_magnitude = np.linalg.norm(gd)
      if g_magnitude > 1e-6:
        gd = [g / g_magnitude for g in gd]
      else:
        gd = [0.0, 0.0, 0.0]
    else:
      gd = [0.0, 0.0, 0.0]

    self.gravity_direction = np.array(
        [gd[2], gd[0], gd[1]],
        np.float32,  # Reorder to z, x, y.
    )
    self.params = params

    # Initializing the injectors.
    self.injectors = [
        injector.injector_factory(ip) for ip in params.lpt.injector
    ]

  def _add_new_particles(
      self,
      lpt_field_ints: lpt_types.LptFieldInts,
      lpt_field_floats: lpt_types.LptFieldFloats,
      new_lpt_field_ints: lpt_types.LptFieldInts,
      new_lpt_field_floats: lpt_types.LptFieldFloats,
  ) -> tuple[lpt_types.LptFieldInts, lpt_types.LptFieldFloats]:
    """Adds particles to this particle field.

    Empty locations marked by `lpt_field_ints[:, 0] == 0` are filled
    with the new particles.

    Args:
      lpt_field_ints: An (n, 2) array of integer particle fields.
      lpt_field_floats: An (n, 7) array of float particle fields.
      new_lpt_field_ints: New particle integer fields to add.
      new_lpt_field_floats: New particle float fields to add.

    Returns:
      Updated (lpt_field_ints, lpt_field_floats) with new particles added.
    """
    particle_status = lpt_field_ints[:, 0]
    n_new_particles = new_lpt_field_ints.shape[0]

    # Determine free locations in the field arrays.
    free_locations = jnp.where(particle_status == 0, size=self.n_max)[0]
    free_locations = free_locations[:n_new_particles]

    # Insert new particles at the free locations.
    lpt_field_ints = lpt_field_ints.at[free_locations].set(new_lpt_field_ints)
    lpt_field_floats = lpt_field_floats.at[free_locations].set(
        new_lpt_field_floats
    )

    return lpt_field_ints, lpt_field_floats

  def _remove_particles(
      self,
      lpt_field_ints: lpt_types.LptFieldInts,
      lpt_field_floats: lpt_types.LptFieldFloats,
      particle_replicas: jax.Array,
  ) -> lpt_types.LptFieldInts:
    """Removes particles that have exited the domain or vaporized.

    Args:
      lpt_field_ints: Integer particle parameters (n, 2).
      lpt_field_floats: Float particle parameters (n, 7).
      particle_replicas: Replica ID of each particle; -1 if out of bounds.

    Returns:
      Updated integer particle parameters with terminated particles deactivated.
    """
    masses = lpt_field_floats[:, lpt_types.COL_MASS]
    should_terminate = (particle_replicas == -1) | (
        masses < self.mass_threshold
    )
    # Set status to 0 for terminated particles.
    new_status = jnp.where(should_terminate, 0, lpt_field_ints[:, 0])
    return lpt_field_ints.at[:, 0].set(new_status)

  def increment_time(
      self,
      replica_id: jax.Array,
      replicas: np.ndarray,
      additional_states: ScalarFieldMap,
      fluid_speeds: jax.Array,
      omegas: jax.Array,
  ) -> tuple[lpt_types.LptFieldInts, lpt_types.LptFieldFloats]:
    """Updates the particles states through time integration.

    Args:
      replica_id: The ID of the replica.
      replicas: A 3D numpy array of replica IDs.
      additional_states: A dictionary including LPT states.
      fluid_speeds: An (n, 3) array of fluid speeds at particle locations.
      omegas: Mass consumption rates for each particle [kg/s].

    Returns:
      Updated (lpt_field_ints, lpt_field_floats).
    """
    lpt_field_ints = additional_states[lpt_types.LPT_INTS_KEY]
    lpt_field_floats = additional_states[lpt_types.LPT_FLOATS_KEY]

    local_min_loc = self._get_local_min_loc(replicas, replica_id)

    def particle_evolution(part_locs, part_vels, part_masses):
      del part_locs, part_masses
      if np.any(self.use_stretched_grid_zxy):
        grid_spacings = self._get_grid_spacings(
            additional_states, local_min_loc
        )
        dxdt = part_vels / grid_spacings
      else:
        dxdt = part_vels
      dvdt = (
          self.c_d / self.tau_p * (fluid_speeds - part_vels)
          + jnp.array(self.gravity_direction) * constants.G
      )
      dmdt = -omegas
      return (dxdt, dvdt, dmdt)

    locs = lpt_field_floats[:, 0:3]
    vels = lpt_field_floats[:, 3:6]
    masses = lpt_field_floats[:, 6]

    locs, vels, masses = time_integration.time_advancement_explicit(
        particle_evolution,
        self.dt,
        numerics_pb2.TimeIntegrationScheme.TIME_SCHEME_RK3,
        (locs, vels, masses),
        (locs, vels, masses),
    )

    lpt_field_floats = jnp.concatenate(
        [locs, vels, masses[:, jnp.newaxis]], axis=1
    )

    return lpt_field_ints, lpt_field_floats

  def _get_local_min_loc(
      self, replicas: np.ndarray, replica_id: jax.Array
  ) -> jax.Array:
    """Returns the local minimum location for each replica including halos.

    Args:
      replicas: A 3D numpy array of replica IDs.
      replica_id: The ID of the replica.

    Returns:
      An array of shape (3,) with the minimum location in z, x, y.
    """
    gp = self.params.grid_params

    # Get core coordinate (x, y, z) from replica_id.
    # replicas has shape (cx, cy, cz).
    coord = np.argwhere(replicas == int(replica_id))[0]
    cx, cy, cz = coord

    min_mapped_loc = (
        jnp.array([cx, cy, cz])
        * jnp.array((gp.core_nx, gp.core_ny, gp.core_nz))
        - gp.halo_width
    ).astype(lpt_types.LPT_FLOAT)

    # Physical min location: compute from grid spacing and core index.
    min_physical_loc = [
        gp.grid_spacings[dim]
        * (
            coord[(dim + 2) % 3] * [gp.core_nx, gp.core_ny, gp.core_nz][dim]
            - gp.halo_width
        )
        for dim in (0, 1, 2)
    ]

    return jnp.stack(
        [
            min_mapped_loc[dim]
            if gp.use_stretched_grid[dim]
            else min_physical_loc[dim]
            for dim in (2, 0, 1)
        ],
        axis=0,
    )

  def _get_grid_spacings(
      self, additional_states: ScalarFieldMap, local_min_loc: jax.Array
  ) -> jax.Array:
    """Returns the grid spacings at the particle locations.

    Args:
      additional_states: A dictionary including LPT states.
      local_min_loc: Minimum location for the replica including halos.

    Returns:
      An (n, 3) array of grid spacings at the particle locations.
    """
    gp = self.params.grid_params
    lpt_field_floats = additional_states[lpt_types.LPT_FLOATS_KEY]
    part_locs = lpt_field_floats[:, :3]

    part_grid_sizes = jnp.zeros((self.n_max, 3))
    for dim in (0, 1, 2):
      dim_xyz = (dim + 2) % 3  # Is zxy for LPT by default.
      if not self.use_stretched_grid_zxy[dim]:
        part_loc_grid_spacings = jnp.ones(
            (self.n_max,), dtype=lpt_types.LPT_FLOAT
        )
      else:
        part_dim_locs = part_locs[:, dim]
        indices = (
            jnp.round(part_dim_locs).astype(lpt_types.LPT_INT)
            - local_min_loc[dim_xyz].astype(lpt_types.LPT_INT)
            + gp.halo_width
        )
        dim_spacings = additional_states[
            stretched_grid_util.h_key(dim_xyz)
        ].reshape(-1)
        indices = jnp.clip(indices, 0, dim_spacings.shape[0] - 1)
        part_loc_grid_spacings = dim_spacings[indices]

      part_grid_sizes = part_grid_sizes.at[:, dim].set(part_loc_grid_spacings)

    return part_grid_sizes

  def step(
      self,
      replica_id: jax.Array,
      replicas: np.ndarray,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      step_id: int,
  ) -> ScalarFieldMap:
    """Updates the particle trajectories and locations.

    Args:
      replica_id: The ID of the replica.
      replicas: A 3D numpy array of replica IDs.
      states: The fluid state fields.
      additional_states: LPT states including positions, velocities, masses.
      step_id: The current time step.

    Returns:
      A dict containing the updated particle states.
    """
    # Inject new particles using user-defined injectors.
    lpt_states = self.inject_particles(
        replica_id, replicas, states, additional_states, step_id, self.params
    )
    additional_states = dict(additional_states)
    additional_states.update(lpt_states)

    return self.update_particles(
        replica_id, replicas, states, additional_states
    )

  def inject_particles(
      self,
      replica_id: jax.Array,
      replicas: np.ndarray,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      step_id: int,
      params: parameters_lib.SwirlLMParameters,
  ) -> dict[str, jax.Array]:
    """Calls each injector region and injects particles into the field.

    Args:
      replica_id: The ID of the replica.
      replicas: A 3D numpy array of replica IDs.
      states: The fluid state fields.
      additional_states: LPT states.
      step_id: The current time step.
      params: The SwirlLMParameters object.

    Returns:
      A dictionary with injected particle field values.
    """
    if params.lpt is None:
      raise ValueError('LPT inject loop called but lpt params are None.')

    new_lpt_ints = jnp.zeros((0, 2), lpt_types.LPT_INT)
    new_lpt_floats = jnp.zeros((0, 7), lpt_types.LPT_FLOAT)
    particles_generated_per_replica = additional_states[
        lpt_types.LPT_COUNTER_KEY
    ]
    for injector_region in self.injectors:
      lpt_states = injector_region.inject(
          replica_id, replicas, states, additional_states, step_id, params
      )

      new_lpt_ints = jnp.concatenate(
          [new_lpt_ints, lpt_states[lpt_types.LPT_INTS_KEY]], axis=0
      )
      new_lpt_floats = jnp.concatenate(
          [new_lpt_floats, lpt_states[lpt_types.LPT_FLOATS_KEY]], axis=0
      )
      particles_generated_per_replica += lpt_states[lpt_types.LPT_COUNTER_KEY]

    # Adding injected particles to the particle field.
    lpt_field_ints = additional_states[lpt_types.LPT_INTS_KEY]
    lpt_field_floats = additional_states[lpt_types.LPT_FLOATS_KEY]
    lpt_field_ints, lpt_field_floats = self._add_new_particles(
        lpt_field_ints, lpt_field_floats, new_lpt_ints, new_lpt_floats
    )
    return {
        lpt_types.LPT_INTS_KEY: lpt_field_ints,
        lpt_types.LPT_FLOATS_KEY: lpt_field_floats,
        lpt_types.LPT_COUNTER_KEY: particles_generated_per_replica,
    }

  @abc.abstractmethod
  def update_particles(
      self,
      replica_id: jax.Array,
      replicas: np.ndarray,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarFieldMap:
    """Moves the particles and communicates with other replicas.

    Args:
      replica_id: The ID of the replica.
      replicas: A 3D numpy array of replica IDs.
      states: The fluid state fields.
      additional_states: LPT states.

    Returns:
      A dict containing the updated particle states.
    """
    pass


def init_fn(params: parameters_lib.SwirlLMParameters) -> dict[str, jax.Array]:
  """Allocates space for the `params.lpt.n_max` particles.

  Args:
    params: The SwirlLMParameters object containing the LPT parameters.

  Returns:
    A dictionary containing zero-declared LPT field arrays.
  """
  if params.lpt is None:
    raise ValueError('LPT init called but lpt params are None.')

  n_max = params.lpt.n_max
  return {
      lpt_types.LPT_INTS_KEY: jnp.zeros((n_max, 2), lpt_types.LPT_INT),
      lpt_types.LPT_FLOATS_KEY: jnp.zeros((n_max, 7), lpt_types.LPT_FLOAT),
      lpt_types.LPT_COUNTER_KEY: jnp.int32(0),
  }


def required_keys(
    lpt_config: lpt_pb2.LagrangianParticleTracking | None,
) -> list[str]:
  """Returns the required keys for the lagrangian particle tracking library."""
  if lpt_config is None:
    return []
  else:
    return [
        lpt_types.LPT_INTS_KEY,
        lpt_types.LPT_FLOATS_KEY,
        lpt_types.LPT_COUNTER_KEY,
    ]
