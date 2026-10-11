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
"""Tests that a simulation does not depend on how it is partitioned.

Running the same physical problem on one device and on several must give the
same answer to round-off. Anything else is a partitioning bug, and the failure
mode is characteristically a discontinuity at a device seam rather than a
crash, so it goes unnoticed without a test like this one.

Needs two devices:
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
      python swirl_lm/jax/core/partitioning_test.py
"""

import re

from absl.testing import absltest
from google.protobuf import text_format
import jax
import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.core import tgv_simulation_test as tgv
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.numerics import convection
from jax.experimental import shard_map
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
import jax.numpy as jnp
from swirl_lm.jax.utility import global_layout
from swirl_lm.jax.utility import grid_parametrization as gp_lib


jax.config.update('jax_enable_x64', False)

# Both runs cover the same global interior, so the per-core extent along the
# split axis differs between them. `kernel_size` must divide every local
# extent, or `ApplyKernelConvOp` silently falls back to a different code path
# and the two runs would no longer be comparable.
_KERNEL_SIZE = 4
_NZ_ONE_DEVICE = 36
_NZ_TWO_DEVICES = 20

_STEPS = 6
_FIELDS = ('u', 'v', 'w', 'p', 'rho')

# Float32 over several steps of a nonlinear solve with an iterative pressure
# solve; the reductions are not associative across a different partitioning.
_RTOL = 1e-4


def _require_two_devices():
  if jax.device_count() < 2:
    raise absltest.SkipTest(
        f'needs 2 devices, found {jax.device_count()}; run with '
        'XLA_FLAGS=--xla_force_host_platform_device_count=2'
    )


def _interiors(params, init_fn, num_steps=_STEPS, fields=_FIELDS):
  """Runs a simulation and returns its global interior, halos stripped."""
  state = driver_lib.run_simulation(
      params=params,
      init_fn=init_fn,
      num_steps=num_steps,
      num_steps_per_cycle=1,
  )
  return {
      key: global_layout.strip_halos_global(
          np.asarray(state[key]), params.grid_params
      )
      for key in fields
  }


def _assert_partition_independent(test, one_device, two_devices, grid_params):
  """Asserts the two runs agree, reporting where along the split axis."""
  seam = grid_params.core_nz - 1
  for key in one_device:
    a, b = one_device[key], two_devices[key]
    test.assertEqual(a.shape, b.shape, f'{key}: global shapes differ')
    scale = max(float(np.max(np.abs(a))), 1e-30)
    diff = np.abs(a - b)
    relative = float(diff.max()) / scale
    worst_z = int(np.argmax(diff.max(axis=(0, 1))))
    test.assertLess(
        relative,
        _RTOL,
        f'{key} depends on the partitioning: max relative difference '
        f'{relative:.3e} at z={worst_z} (the device seam is between '
        f'z={seam} and z={seam + 1}).',
    )


class TgvPartitioningTest(absltest.TestCase):
  """Control: a periodic problem with no boundary models."""

  def test_tgv_is_partition_independent(self):
    _require_two_devices()

    def params(cz, nz):
      text = tgv._DRIVER_TGV_CONFIG_111  # pylint: disable=protected-access
      text = re.sub(
          r'computation_shape \{ dim_0: 1 dim_1: 1 dim_2: 1 \}',
          f'computation_shape {{ dim_0: 1 dim_1: 1 dim_2: {cz} }}',
          text,
      )
      text = re.sub(
          r'grid_size \{ dim_0: 32 dim_1: 32 dim_2: 32 \}',
          f'grid_size {{ dim_0: 32 dim_1: 32 dim_2: {nz} }}',
          text,
      )
      text = re.sub(r'kernel_size: \d+', f'kernel_size: {_KERNEL_SIZE}', text)
      config = parameters_pb2.SwirlLMParameters()
      text_format.Parse(text, config)
      return parameters_lib.SwirlLMParameters(config)

    one = params(1, _NZ_ONE_DEVICE)
    two = params(2, _NZ_TWO_DEVICES)
    self.assertEqual(
        one.grid_params.cz * one.grid_params.core_nz,
        two.grid_params.cz * two.grid_params.core_nz,
    )

    _assert_partition_independent(
        self,
        _interiors(one, tgv._tgv_driver_init_fn),  # pylint: disable=protected-access
        _interiors(two, tgv._tgv_driver_init_fn),  # pylint: disable=protected-access
        two.grid_params,
    )


class FirePartitioningTest(absltest.TestCase):
  """The real case: inflow-driven, with per-replica 2D inflow planes.

  This is the configuration whose z-split produced a 7.7 m/s step in `u` at
  the device seam, because the per-replica `mean_*` inflow planes were
  replicated from replica 0.

  The config is taken from the example but every field this test depends on is
  overridden on the parsed proto rather than by rewriting its text, so editing
  `fire.pbtxt` cannot silently change what is being tested.
  """

  def _params_and_sim(self, cz, nz):
    # Imported here so the TGV control and the unit tests below do not depend
    # on the example.
    from swirl_lm.jax.examples.fire import fire  # pylint: disable=g-import-not-at-top

    config = parameters_pb2.SwirlLMParameters()
    with open('swirl_lm/jax/examples/fire/fire.pbtxt', 'r') as f:
      text_format.Parse(f.read(), config)

    grid = config.grid_params
    grid.computation_shape.dim_0 = 1
    grid.computation_shape.dim_1 = 1
    grid.computation_shape.dim_2 = cz
    grid.grid_size.dim_0 = 16
    grid.grid_size.dim_1 = 16
    grid.grid_size.dim_2 = nz
    grid.kernel_size = _KERNEL_SIZE
    # Flat terrain and no surface model, so this isolates the inflow: with
    # those enabled a smaller, separate seam term remains.
    config.boundary_models.ClearField('ib')
    config.boundary_models.ClearField('most')
    for key in ('ib_interior_mask', 'ib_boundary'):
      if key in config.additional_state_keys:
        config.additional_state_keys.remove(key)

    params = parameters_lib.SwirlLMParameters(config)
    simulation = fire.Fire(
        params,
        # Zero fluctuation amplitude: the synthetic inflow draws random fields
        # sized by the per-core mesh, which differs between the two runs, so
        # the turbulence itself cannot be compared. The mean profile, which is
        # what the bug corrupted, is unaffected.
        u_rms=0.0,
        v_rms=0.0,
        w_rms=0.0,
        # A short integral length scale keeps the inflow's digital filter
        # narrow enough for this small per-core grid; the default of 10 cells
        # needs at least 20 cells per core in each in-plane direction.
        inflow_length_scale_cells=2.0,
    )
    params.source_update_fn_lib = simulation.source_term_update_fn()
    params.additional_states_update_fn = (
        simulation.additional_states_update_fn
    )
    params.preprocessing_states_update_fn = (
        simulation.pre_simulation_update_fn
    )
    return params, simulation

  def test_fire_is_partition_independent(self):
    _require_two_devices()
    one_params, one_sim = self._params_and_sim(1, _NZ_ONE_DEVICE)
    two_params, two_sim = self._params_and_sim(2, _NZ_TWO_DEVICES)
    self.assertEqual(
        one_params.grid_params.cz * one_params.grid_params.core_nz,
        two_params.grid_params.cz * two_params.grid_params.core_nz,
    )

    fields = ('u', 'w', 'p', 'theta')
    _assert_partition_independent(
        self,
        _interiors(one_params, one_sim.initialization, fields=fields),
        _interiors(two_params, two_sim.initialization, fields=fields),
        two_params.grid_params,
    )


# A wall-bounded channel: periodic in x and y, a free-slip wall at z-low and a
# Neumann outflow at z-high. This is the boundary layout of the fire case, and
# it is what the all-periodic TGV control cannot reach -- `find_bc_type` returns
# PERIODIC for every face there, so the physical-boundary code paths in the
# convection scheme are never entered.
_WALL_CHANNEL_CONFIG = """\
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {
  computation_shape { dim_0: 1 dim_1: 1 dim_2: %(cz)d }
  grid_size { dim_0: 16 dim_1: 16 dim_2: %(nz)d }
  length { dim_0: 1.0 dim_1: 1.0 dim_2: 2.0 }
  halo_width: 2
  dt: 1e-3
  kernel_size: %(kernel)d
  periodic { dim_0: true dim_1: true dim_2: false }
}
pressure {
  solver { jacobi { max_iterations: 5 halo_width: 2 omega: 0.67 } }
}
thermodynamics { constant_density {} }
density: 1.0
kinematic_viscosity: 6.25e-4
num_sub_iterations: 3
use_sgs: false
enable_rhie_chow_correction: false
boundary_conditions {
  name: "u"
  boundary_info { dim: 2 location: 0 type: BC_TYPE_NEUMANN value: 0.0 }
  boundary_info { dim: 2 location: 1 type: BC_TYPE_NEUMANN value: 0.0 }
}
boundary_conditions {
  name: "v"
  boundary_info { dim: 2 location: 0 type: BC_TYPE_NEUMANN value: 0.0 }
  boundary_info { dim: 2 location: 1 type: BC_TYPE_NEUMANN value: 0.0 }
}
boundary_conditions {
  name: "w"
  boundary_info { dim: 2 location: 0 type: BC_TYPE_DIRICHLET value: 0.0 }
  boundary_info { dim: 2 location: 1 type: BC_TYPE_NEUMANN value: 0.0 }
}
boundary_conditions {
  name: "p"
  boundary_info { dim: 2 location: 0 type: BC_TYPE_NEUMANN value: 0.0 }
  boundary_info { dim: 2 location: 1 type: BC_TYPE_NEUMANN value: 0.0 }
}
"""

_WALL_NZ_ONE_DEVICE = 36
_WALL_NZ_TWO_DEVICES = 20


def _wall_channel_init_fn(params, logical_coordinates):
  """A rising flow in a channel, identical in the 1- and 2-device layouts.

  The field is written from *global* coordinates so both partitionings start
  from the same physical state. A strong mean `w` is what makes the test
  sensitive: an erroneously blocked wall-normal flux at a device seam is a
  first-order error where `w` is large, and only round-off where it is small.
  """
  gp = params.grid_params
  hw = gp.halo_width
  nx, ny, nz = gp.nx, gp.ny, gp.nz
  dx, dy, dz = gp.to_xyz_order(gp.grid_spacings)

  # Global index of this device's first point, halos included.
  z0 = logical_coordinates[2] * (nz - 2 * hw) - hw
  x = (np.arange(nx) - hw + 0.5) * dx
  y = (np.arange(ny) - hw + 0.5) * dy
  z = (np.arange(z0, z0 + nz) + 0.5) * dz
  xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

  two_pi = 2.0 * np.pi
  return {
      'rho': jnp.ones((nx, ny, nz), jnp.float32),
      'u': jnp.asarray(0.2 * np.sin(two_pi * xx) * np.cos(two_pi * yy), jnp.float32),
      'v': jnp.asarray(0.2 * np.cos(two_pi * xx) * np.sin(two_pi * yy), jnp.float32),
      'w': jnp.asarray(1.0 + 0.1 * np.sin(two_pi * xx) * np.sin(two_pi * yy), jnp.float32),
      'p': jnp.zeros((nx, ny, nz), jnp.float32),
  }


class WallChannelPartitioningTest(absltest.TestCase):
  """A non-periodic problem, so the physical-boundary paths are exercised.

  `TgvPartitioningTest` is periodic in all three dimensions and therefore
  structurally cannot detect a defect in physical-boundary handling. This case
  puts a free-slip wall on the z-low face and splits along z, so the only wall
  is on device 0 and every other z face at a device boundary is an interior
  seam that must stay transparent.
  """

  def _params(self, cz, nz):
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(
        _WALL_CHANNEL_CONFIG
        % {'cz': cz, 'nz': nz, 'kernel': _KERNEL_SIZE},
        config,
    )
    return parameters_lib.SwirlLMParameters(config)

  def test_z_low_is_a_slip_wall(self):
    # Guards the premise: if this ever stops classifying as a wall, the test
    # below silently stops covering the code path it was written for.
    params = self._params(2, _WALL_NZ_TWO_DEVICES)
    self.assertEqual(
        params.bc_type[2][0], boundary_condition_utils.BoundaryType.SLIP_WALL
    )
    self.assertNotIn(
        params.bc_type[2][1],
        convection._WALL_TYPES,  # pylint: disable=protected-access
    )

  def test_wall_channel_is_partition_independent(self):
    _require_two_devices()
    one = self._params(1, _WALL_NZ_ONE_DEVICE)
    two = self._params(2, _WALL_NZ_TWO_DEVICES)
    self.assertEqual(
        one.grid_params.cz * one.grid_params.core_nz,
        two.grid_params.cz * two.grid_params.core_nz,
    )

    _assert_partition_independent(
        self,
        _interiors(one, _wall_channel_init_fn),
        _interiors(two, _wall_channel_init_fn),
        two.grid_params,
    )


class WallNormalFluxTest(absltest.TestCase):
  """A physical wall BC may only be imposed by the device that owns it.

  `_zero_wall_normal_flux` enforces impermeability by zeroing the wall-normal
  face flux at a fixed index of the array it is given. Inside `shard_map` that
  array is one device's subdomain, so without a boundary-device check every
  device zeroes the flux at its own local face -- blocking transport across
  every interior seam. On the 4-GPU fire case that produced a persistent
  ~0.7 m/s jump in `u` at each of the three device boundaries.
  """

  def _run_sharded(self, cz, bc_types):
    """Applies the flux zeroing under shard_map and returns the global result."""
    text = f"""
    computation_shape {{ dim_x: 1 dim_y: 1 dim_z: {cz} }}
    grid_size {{ dim_x: 8 dim_y: 8 dim_z: 8 }}
    length {{ dim_x: 1.0 dim_y: 1.0 dim_z: 1.0 }}
    halo_width: 2
    dt: 0.1
    data_axis_order: 'xyz'
    """
    grid_params = gp_lib.GridParametrization(
        gp_lib.params_from_text_proto(text)
    )
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(
        'grid_params {'
        f'  computation_shape {{ dim_0: 1 dim_1: 1 dim_2: {cz} }}'
        '  grid_size { dim_0: 8 dim_1: 8 dim_2: 8 }'
        '  length { dim_0: 1.0 dim_1: 1.0 dim_2: 1.0 }'
        '  halo_width: 2 dt: 0.1'
        '}',
        config,
    )
    params = parameters_lib.SwirlLMParameters(config)
    mesh = driver_lib.create_mesh(params)

    # A flux of 1 everywhere, so any zeroed plane stands out.
    flux = jnp.ones((8, 8, 8 * cz), jnp.float32)

    def body(f):
      return convection._zero_wall_normal_flux(  # pylint: disable=protected-access
          f, 'z', 'w', bc_types, grid_params.halo_width, grid_params
      )

    spec = P('x', 'y', 'z')
    fn = jax.jit(
        shard_map.shard_map(
            body, mesh=mesh, in_specs=(spec,), out_specs=spec, check_rep=False
        )
    )
    return np.asarray(fn(flux))

  def test_wall_flux_is_zeroed_on_one_device(self):
    _require_two_devices()
    wall = boundary_condition_utils.BoundaryType.SLIP_WALL
    neumann = boundary_condition_utils.BoundaryType.OUTFLOW
    out = self._run_sharded(1, (wall, neumann))

    # halo_width == 2, so the wall face sits at index 2.
    np.testing.assert_array_equal(out[:, :, 2], np.zeros((8, 8), np.float32))
    self.assertTrue((out[:, :, 3:] == 1.0).all())

  def test_interior_seams_are_not_treated_as_walls(self):
    _require_two_devices()
    wall = boundary_condition_utils.BoundaryType.SLIP_WALL
    neumann = boundary_condition_utils.BoundaryType.OUTFLOW
    out = self._run_sharded(2, (wall, neumann))

    # Device 0 owns the physical wall: its local index 2 must be zeroed.
    np.testing.assert_array_equal(out[:, :, 2], np.zeros((8, 8), np.float32))
    # Device 1's local low face is global index 8 + 2 = 10. That is an interior
    # seam, and zeroing it would block vertical transport between subdomains.
    self.assertTrue(
        (out[:, :, 10] == 1.0).all(),
        'the flux was zeroed at an interior device seam, not just at the wall',
    )
    # Nothing else is touched.
    kept = [z for z in range(16) if z != 2]
    self.assertTrue((out[:, :, kept] == 1.0).all())

  def test_non_wall_boundary_is_left_alone(self):
    _require_two_devices()
    neumann = boundary_condition_utils.BoundaryType.OUTFLOW
    out = self._run_sharded(2, (neumann, neumann))
    self.assertTrue((out == 1.0).all())


class DistributeStatesTest(absltest.TestCase):
  """Unit coverage for the per-key mapping of states onto the mesh."""

  def _mesh(self, cz=2):
    text = f"""
    computation_shape {{ dim_x: 1 dim_y: 1 dim_z: {cz} }}
    grid_size {{ dim_x: 8 dim_y: 8 dim_z: 8 }}
    length {{ dim_x: 1.0 dim_y: 1.0 dim_z: 1.0 }}
    halo_width: 2
    dt: 0.1
    data_axis_order: 'xyz'
    """
    grid_params = gp_lib.GridParametrization(
        gp_lib.params_from_text_proto(text)
    )
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(
        'grid_params {'
        f'  computation_shape {{ dim_0: 1 dim_1: 1 dim_2: {cz} }}'
        '  grid_size { dim_0: 8 dim_1: 8 dim_2: 8 }'
        '  length { dim_0: 1.0 dim_1: 1.0 dim_2: 1.0 }'
        '  halo_width: 2 dt: 0.1'
        '}',
        config,
    )
    params = parameters_lib.SwirlLMParameters(config)
    return params, driver_lib.create_mesh(params), grid_params

  def test_3d_fields_keep_their_own_subdomain(self):
    _require_two_devices()
    params, mesh, grid_params = self._mesh()
    per_replica = [
        {'u': np.full((8, 8, 8), float(i), np.float32)} for i in range(2)
    ]

    state = driver_lib._distribute_states(  # pylint: disable=protected-access
        per_replica, mesh, grid_params, params=params
    )

    assembled = np.asarray(state['u'])
    self.assertEqual(assembled.shape, (8, 8, 16))
    # Each replica's tile must land on its own half, not be broadcast.
    np.testing.assert_array_equal(assembled[:, :, :8], per_replica[0]['u'])
    np.testing.assert_array_equal(assembled[:, :, 8:], per_replica[1]['u'])

  def test_replica_invariant_low_rank_state_is_replicated(self):
    _require_two_devices()
    params, mesh, grid_params = self._mesh()
    per_replica = [{'simulation_time': np.float32(0.5)} for _ in range(2)]

    state = driver_lib._distribute_states(  # pylint: disable=protected-access
        per_replica, mesh, grid_params, params=params
    )

    self.assertEqual(state['simulation_time'].shape, ())
    self.assertEqual(float(state['simulation_time']), 0.5)

  def test_per_replica_low_rank_state_without_a_spec_raises(self):
    """The bug that produced a seam: replica 0's plane silently winning."""
    _require_two_devices()
    params, mesh, grid_params = self._mesh()
    per_replica = [
        {'mean_u_0_0': np.full((8, 8), float(i), np.float32)}
        for i in range(2)
    ]

    with self.assertRaisesRegex(ValueError, 'differs between replicas'):
      driver_lib._distribute_states(  # pylint: disable=protected-access
          per_replica, mesh, grid_params, params=params
      )

  def test_per_replica_low_rank_state_with_a_spec_is_distributed(self):
    _require_two_devices()
    params, mesh, grid_params = self._mesh()
    params.additional_state_partition_specs['mean_u_0_0'] = (
        jax.sharding.PartitionSpec('y', 'z')
    )
    per_replica = [
        {'mean_u_0_0': np.full((8, 8), float(i), np.float32)}
        for i in range(2)
    ]

    state = driver_lib._distribute_states(  # pylint: disable=protected-access
        per_replica, mesh, grid_params, params=params
    )

    assembled = np.asarray(state['mean_u_0_0'])
    self.assertEqual(assembled.shape, (8, 16))
    np.testing.assert_array_equal(assembled[:, :8], per_replica[0]['mean_u_0_0'])
    np.testing.assert_array_equal(assembled[:, 8:], per_replica[1]['mean_u_0_0'])


if __name__ == '__main__':
  absltest.main()
