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
"""Integration test: Taylor-Green Vortex on the JAX solver.

This test verifies the full simulation pipeline end-to-end:
  - Config parsing from textproto.
  - SwirlLMParameters construction.
  - TGV initialization.
  - Predictor-corrector time stepping (simulation.step).
  - Basic physical validation: kinetic energy decays monotonically.

The simulation step must run inside `shard_map` because the halo exchange
uses `jax.lax.axis_index`, which requires a named-axis context.
The step function is JIT-compiled for performance.
"""


import itertools
import os
import tempfile

from absl.testing import absltest
from google.protobuf import text_format
import jax
from jax.experimental import shard_map
import jax.experimental.mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.core import simulation as simulation_lib
from swirl_lm.jax.equations import common
from swirl_lm.jax.io import visualization as visualization_lib
from swirl_lm.jax.utility import stretched_grid
from swirl_lm.jax.utility import stretched_grid_util

jax.config.update('jax_enable_x64', False)

# TGV config textproto — minimal config sized for fast testing.
# Uses KERNEL_OP_SLICE to avoid kernel_size divisibility constraints, and
# a small 8x8x8 grid with few Jacobi iterations for speed.
_TGV_CONFIG = """
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {
  computation_shape { dim_0: 1 dim_1: 1 dim_2: 1 }
  grid_size { dim_0: 8 dim_1: 8 dim_2: 8 }
  length { dim_0: 6.2831853 dim_1: 6.2831853 dim_2: 6.2831853 }
  halo_width: 2
  dt: 1e-3
  kernel_size: 12
  periodic { dim_0: true dim_1: true dim_2: true }
}
pressure {
  solver {
    jacobi {
      max_iterations: 5 halo_width: 2 omega: 0.67
    }
  }
}
thermodynamics {
  constant_density {}
}
density: 1.0
kinematic_viscosity: 6.25e-4
num_sub_iterations: 3
use_sgs: false
enable_rhie_chow_correction: false
"""

_U_MAG = 1.0
_RHO = 1.0

_STATE_KEYS = (
    common.KEY_RHO,
    common.KEY_U,
    common.KEY_V,
    common.KEY_W,
    common.KEY_P,
)


def _make_params() -> parameters_lib.SwirlLMParameters:
  """Creates SwirlLMParameters from inline textproto."""
  config = parameters_pb2.SwirlLMParameters()
  text_format.Parse(_TGV_CONFIG, config)
  return parameters_lib.SwirlLMParameters(config)


def _tgv_init(
    params: parameters_lib.SwirlLMParameters,
) -> dict[str, jax.Array]:
  """Initializes the Taylor-Green Vortex flow field.

  u =  U sin(x) cos(y) cos(z)
  v = -U cos(x) sin(y) cos(z)
  w = 0
  p = (rho U^2 / 16) * (cos(2z) + 2) * (cos(2x) + cos(2y))

  Args:
    params: Simulation parameters.

  Returns:
    Initial states dict with rho, u, v, w, p.
  """
  gp = params.grid_params
  nx = gp.nx + 2 * gp.halo_width
  ny = gp.ny + 2 * gp.halo_width
  nz = gp.nz + 2 * gp.halo_width

  dx = gp.to_xyz_order(gp.grid_spacings)[0]
  dy = gp.to_xyz_order(gp.grid_spacings)[1]
  dz = gp.to_xyz_order(gp.grid_spacings)[2]

  x_1d = jnp.arange(nx) * dx - gp.halo_width * dx
  y_1d = jnp.arange(ny) * dy - gp.halo_width * dy
  z_1d = jnp.arange(nz) * dz - gp.halo_width * dz

  xx, yy, zz = jnp.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

  u = _U_MAG * jnp.sin(xx) * jnp.cos(yy) * jnp.cos(zz)
  v = -_U_MAG * jnp.cos(xx) * jnp.sin(yy) * jnp.cos(zz)
  w = jnp.zeros_like(xx)
  p = (
      _RHO
      * _U_MAG**2
      / 16.0
      * ((jnp.cos(2.0 * zz) + 2.0) * (jnp.cos(2.0 * xx) + jnp.cos(2.0 * yy)))
  )
  rho = _RHO * jnp.ones_like(xx)

  return {
      common.KEY_RHO: rho,
      common.KEY_U: u,
      common.KEY_V: v,
      common.KEY_W: w,
      common.KEY_P: p,
  }


def _kinetic_energy(states: dict[str, jax.Array], hw: int = 2) -> float:
  """Computes mean kinetic energy over the interior (excluding halos)."""
  s = slice(hw, -hw) if hw > 0 else slice(None)
  u = states[common.KEY_U][s, s, s]
  v = states[common.KEY_V][s, s, s]
  w = states[common.KEY_W][s, s, s]
  rho = states[common.KEY_RHO][s, s, s]
  ke = 0.5 * rho * (u**2 + v**2 + w**2)
  return float(jnp.mean(ke))


def _make_step_fn(
    sim: simulation_lib.Simulation,
    mesh: Mesh,
):
  """Creates a JIT-compiled, shard_map-wrapped step function.

  The simulation step calls `jax.lax.axis_index` inside halo exchange,
  which requires a named-axis context provided by `shard_map`.
  JIT compilation is applied on top for performance.

  Args:
    sim: The simulation object.
    mesh: JAX mesh for device topology.

  Returns:
    A JIT-compiled function that takes and returns state dicts.
  """
  spec = P(*mesh.axis_names)
  in_specs = {k: spec for k in _STATE_KEYS}
  out_specs = {k: spec for k in _STATE_KEYS}

  def step_fn(states):
    return sim.step(states, {}, mesh)

  return jax.jit(
      shard_map.shard_map(
          step_fn,
          mesh=mesh,
          in_specs=(in_specs,),
          out_specs=out_specs,
          check_rep=False,
      )
  )


class TgvSimulationTest(absltest.TestCase):
  """Integration tests for TGV on the JAX solver."""

  def _setup(self):
    """Creates params, mesh, init states, and JIT-compiled step function."""
    params = _make_params()
    devices = jax.experimental.mesh_utils.create_device_mesh(
        (1, 1, 1), devices=jax.local_devices()[:1]
    )
    mesh = Mesh(devices, axis_names=params.grid_params.data_axis_order)
    states = _tgv_init(params)
    sim = simulation_lib.Simulation(params)
    step_fn = _make_step_fn(sim, mesh)
    return states, step_fn

  def test_simulation_runs_without_error(self):
    """Verifies the simulation can run a few steps without crashing."""
    states, step_fn = self._setup()

    # Run 2 steps (first step includes JIT compilation time).
    for _ in range(2):
      states = step_fn(states)

    # Verify output has the expected keys.
    for key in _STATE_KEYS:
      self.assertIn(key, states)

  def test_kinetic_energy_decays(self):
    """Verifies kinetic energy decays for viscous TGV flow."""
    states, step_fn = self._setup()
    ke_initial = _kinetic_energy(states)

    # Run 3 steps.
    for _ in range(3):
      states = step_fn(states)

    ke_final = _kinetic_energy(states)

    # Kinetic energy must decrease for viscous flow.
    self.assertGreater(ke_initial, 0.0)
    self.assertLess(
        ke_final,
        ke_initial,
        f'KE should decay: initial={ke_initial}, final={ke_final}',
    )

  def test_density_stays_constant(self):
    """For constant-density TGV, rho should not change."""
    states, step_fn = self._setup()

    for _ in range(2):
      states = step_fn(states)

    # Density should remain constant.
    np.testing.assert_allclose(
        states[common.KEY_RHO],
        _RHO * jnp.ones_like(states[common.KEY_RHO]),
        atol=1e-10,
    )


def _make_periodic_stretched_coord_file(
    n_points: int,
    domain_size: float,
    stretch_amplitude: float = 0.1,
) -> tuple[str, np.ndarray]:
  """Creates a temp file with mildly stretched periodic coordinates.

  Generates a sinusoidally perturbed uniform grid: x_i = i*dx + A*sin(2*pi*i/N).
  The final point (= first point + domain_size) is appended for the periodic
  convention expected by grid_parametrization.

  Args:
    n_points: Number of interior grid points.
    domain_size: Domain length.
    stretch_amplitude: Amplitude of sinusoidal perturbation.

  Returns:
    Tuple of (filepath, coordinate_array_without_final_point).
  """
  dx = domain_size / n_points
  indices = np.arange(n_points)
  coords = indices * dx + stretch_amplitude * np.sin(
      2.0 * np.pi * indices / n_points
  )
  # Append final point for periodic convention.
  coords_with_end = np.append(coords, domain_size)

  tmpdir = tempfile.mkdtemp()
  filepath = os.path.join(tmpdir, 'stretched_coords.txt')
  np.savetxt(filepath, coords_with_end, fmt='%.15f')
  return filepath, coords


# Stretched grid TGV config template. The dim_z path is filled at runtime.
_TGV_STRETCHED_CONFIG_TEMPLATE = """\
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {{
  computation_shape {{ dim_0: 1 dim_1: 1 dim_2: 1 }}
  grid_size {{ dim_0: 8 dim_1: 8 dim_2: 8 }}
  length {{ dim_0: 6.2831853 dim_1: 6.2831853 dim_2: 0.0 }}
  halo_width: 2
  dt: 1e-3
  kernel_size: 8
  periodic {{ dim_0: true dim_1: true dim_2: true }}
  stretched_grid_files {{ dim_2 {{ path: "{coord_file}" }} }}
}}
pressure {{
  solver {{
    jacobi {{
      max_iterations: 5 halo_width: 2 omega: 0.67
    }}
  }}
}}
thermodynamics {{
  constant_density {{}}
}}
density: 1.0
kinematic_viscosity: 6.25e-4
num_sub_iterations: 3
use_sgs: false
enable_rhie_chow_correction: false
"""


class TgvStretchedGridTest(absltest.TestCase):
  """Integration tests for TGV with a stretched grid in one dimension."""

  def _setup(self):
    """Creates params with stretched z-grid, mesh, states, and step fn."""
    domain_z = 2.0 * np.pi
    coord_file, _ = _make_periodic_stretched_coord_file(
        n_points=4, domain_size=domain_z, stretch_amplitude=0.05
    )

    config_text = _TGV_STRETCHED_CONFIG_TEMPLATE.format(coord_file=coord_file)
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(config_text, config)
    params = parameters_lib.SwirlLMParameters(config)

    devices = jax.experimental.mesh_utils.create_device_mesh(
        (1, 1, 1), devices=jax.local_devices()[:1]
    )
    mesh = Mesh(devices, axis_names=params.grid_params.data_axis_order)

    # Initialize flow field using gp.nx (= grid_size from proto), which
    # already includes halos: core_n + 2*halo_width. This matches the size of
    # the stretched grid scale factors from
    # local_stretched_grid_vars_from_global_xyz.
    gp = params.grid_params
    nx, ny, _ = gp.nx, gp.ny, gp.nz
    dx = gp.to_xyz_order(gp.grid_spacings)[0]
    dy = gp.to_xyz_order(gp.grid_spacings)[1]

    # For the stretched z-dimension, use global_xyz_with_halos coordinates.
    z_coords = gp.global_xyz_with_halos[gp.get_axis_index('z')]

    x_1d = jnp.arange(nx) * dx - gp.halo_width * dx
    y_1d = jnp.arange(ny) * dy - gp.halo_width * dy
    z_1d = z_coords

    xx, yy, zz = jnp.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

    states = {
        common.KEY_RHO: _RHO * jnp.ones_like(xx),
        common.KEY_U: _U_MAG * jnp.sin(xx) * jnp.cos(yy) * jnp.cos(zz),
        common.KEY_V: -_U_MAG * jnp.cos(xx) * jnp.sin(yy) * jnp.cos(zz),
        common.KEY_W: jnp.zeros_like(xx),
        common.KEY_P: (
            _RHO
            * _U_MAG**2
            / 16.0
            * (
                (jnp.cos(2.0 * zz) + 2.0)
                * (jnp.cos(2.0 * xx) + jnp.cos(2.0 * yy))
            )
        ),
    }

    # Compute stretched grid scale factors.
    sg_vars = stretched_grid.local_stretched_grid_vars_from_global_xyz(
        params, logical_coordinates=(0, 0, 0)
    )

    sim = simulation_lib.Simulation(params)

    # Determine all state keys for shard_map specs.
    all_state_keys = list(_STATE_KEYS)
    additional_keys = list(sg_vars.keys())
    all_keys = all_state_keys + additional_keys

    spec = P(*mesh.axis_names)
    in_specs = {k: spec for k in all_keys}
    out_specs = {k: spec for k in all_keys}

    # Separate essential and additional states for the step call.
    def step_fn(combined_states):
      ess = {k: combined_states[k] for k in _STATE_KEYS}
      add = {k: combined_states[k] for k in additional_keys}
      return sim.step(ess, add, mesh)

    jit_step = jax.jit(
        shard_map.shard_map(
            step_fn,
            mesh=mesh,
            in_specs=(in_specs,),
            out_specs=out_specs,
            check_rep=False,
        )
    )

    # Merge states and stretched grid vars for combined input.
    combined = dict(states)
    combined.update(sg_vars)

    return combined, jit_step, all_state_keys, additional_keys

  def test_stretched_grid_init_produces_valid_states(self):
    """Verifies stretched grid initialization produces valid scale factors."""
    combined, _, _, additional_keys = self._setup()

    # Stretched grid should produce h and h_face keys for the z dimension.
    z_dim = 2  # z is the stretched dimension
    h_key = stretched_grid_util.h_key(z_dim)
    h_face_key = stretched_grid_util.h_face_key(z_dim)
    self.assertIn(h_key, additional_keys)
    self.assertIn(h_face_key, additional_keys)

    # Scale factors should be positive (physical grid spacing > 0).
    h = combined[h_key]
    h_face = combined[h_face_key]
    self.assertTrue(jnp.all(h > 0), f'h has non-positive values: {h}')
    self.assertTrue(
        jnp.all(h_face > 0), f'h_face has non-positive values: {h_face}'
    )

    # Scale factors should be broadcastable to 3D field shape.
    rho = combined[common.KEY_RHO]
    # h has shape (1, 1, nz) for dim=2, should broadcast to (nx, ny, nz).
    result = rho * h  # Should not raise.
    self.assertEqual(result.shape, rho.shape)

  def test_stretched_grid_simulation_runs(self):
    """Verifies simulation with stretched grid runs without error."""
    combined, jit_step, _, additional_keys = self._setup()

    for _ in range(2):
      result = jit_step(combined)
      # Carry over additional_states (unchanged by step).
      combined = dict(result)
      for k in additional_keys:
        if k not in combined:
          combined[k] = self._setup()[0][k]

    for key in _STATE_KEYS:
      self.assertIn(key, result)

  def test_stretched_grid_kinetic_energy_decays(self):
    """Verifies KE decays for stretched-grid TGV."""
    combined, jit_step, _, additional_keys = self._setup()
    ke_initial = _kinetic_energy(combined)

    for _ in range(3):
      result = jit_step(combined)
      combined = dict(result)
      for k in additional_keys:
        if k not in combined:
          combined[k] = self._setup()[0][k]

    ke_final = _kinetic_energy(result)

    self.assertGreater(ke_initial, 0.0)
    self.assertLess(
        ke_final,
        ke_initial,
        f'KE should decay: initial={ke_initial}, final={ke_final}',
    )


# ---- Driver-level TGV tests (using run_simulation) ---- #

# Config for driver tests: single device, periodic TGV.
_DRIVER_TGV_CONFIG_111 = """\
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {
  computation_shape { dim_0: 1 dim_1: 1 dim_2: 1 }
  grid_size { dim_0: 32 dim_1: 32 dim_2: 32 }
  length { dim_0: 6.2831853 dim_1: 6.2831853 dim_2: 6.2831853 }
  halo_width: 2
  dt: 1e-3
  kernel_size: 8
  periodic { dim_0: true dim_1: true dim_2: true }
}
pressure {
  solver {
    jacobi {
      max_iterations: 5 halo_width: 2 omega: 0.67
    }
  }
}
thermodynamics {
  constant_density {}
}
density: 1.0
kinematic_viscosity: 6.25e-4
num_sub_iterations: 3
use_sgs: false
enable_rhie_chow_correction: false
"""

# Config for driver tests: 4 devices (2x2x1), periodic TGV.
_DRIVER_TGV_CONFIG_221 = """\
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {
  computation_shape { dim_0: 2 dim_1: 2 dim_2: 1 }
  grid_size { dim_0: 16 dim_1: 16 dim_2: 32 }
  length { dim_0: 6.2831853 dim_1: 6.2831853 dim_2: 6.2831853 }
  halo_width: 2
  dt: 1e-3
  kernel_size: 8
  periodic { dim_0: true dim_1: true dim_2: true }
}
pressure {
  solver {
    jacobi {
      max_iterations: 5 halo_width: 2 omega: 0.67
    }
  }
}
thermodynamics {
  constant_density {}
}
density: 1.0
kinematic_viscosity: 6.25e-4
num_sub_iterations: 3
use_sgs: false
enable_rhie_chow_correction: false
"""


def _tgv_driver_init_fn(
    params: parameters_lib.SwirlLMParameters,
    logical_coordinates: tuple[int, int, int],
) -> dict[str, jax.Array]:
  """TGV init function matching the driver's InitFn signature.

  Generates device-local TGV initial conditions based on the device's
  position in the computational grid (logical_coordinates).

  Args:
    params: Simulation parameters.
    logical_coordinates: (cx_i, cy_i, cz_i) position of this device.

  Returns:
    Device-local initial states dict with rho, u, v, w, p.
  """
  gp = params.grid_params
  hw = gp.halo_width

  # Core points per device in each dimension.
  core_nx = gp.nx - 2 * hw
  core_ny = gp.ny - 2 * hw
  core_nz = gp.nz - 2 * hw

  # Total points per device (core + halos).
  local_nx = core_nx + 2 * hw
  local_ny = core_ny + 2 * hw
  local_nz = core_nz + 2 * hw

  # Grid spacing (uniform).
  dx, dy, dz = gp.to_xyz_order(gp.grid_spacings)

  # Global coordinate offsets for this device.
  cx_i, cy_i, cz_i = logical_coordinates
  x_offset = cx_i * core_nx * dx
  y_offset = cy_i * core_ny * dy
  z_offset = cz_i * core_nz * dz

  # Local coordinates including halos.
  x_1d = jnp.arange(local_nx) * dx - hw * dx + x_offset
  y_1d = jnp.arange(local_ny) * dy - hw * dy + y_offset
  z_1d = jnp.arange(local_nz) * dz - hw * dz + z_offset

  xx, yy, zz = jnp.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

  u = _U_MAG * jnp.sin(xx) * jnp.cos(yy) * jnp.cos(zz)
  v = -_U_MAG * jnp.cos(xx) * jnp.sin(yy) * jnp.cos(zz)
  w = jnp.zeros_like(xx)
  p = (
      _RHO
      * _U_MAG**2
      / 16.0
      * ((jnp.cos(2.0 * zz) + 2.0) * (jnp.cos(2.0 * xx) + jnp.cos(2.0 * yy)))
  )
  rho = _RHO * jnp.ones_like(xx)

  return {
      common.KEY_RHO: rho,
      common.KEY_U: u,
      common.KEY_V: v,
      common.KEY_W: w,
      common.KEY_P: p,
  }


def _remove_halos(params, states, varnames):
  gp = params.grid_params
  hw = gp.halo_width

  states_no_halos = {}
  for key, v in states.items():
    if key not in varnames:
      continue
    buf = np.zeros((gp.fx, gp.fy, gp.fz), dtype=v.dtype)
    for i, j, k in itertools.product(range(gp.cx), range(gp.cy), range(gp.cz)):
      buf[
          i * gp.core_nx : (i + 1) * gp.core_nx,
          j * gp.core_ny : (j + 1) * gp.core_ny,
          k * gp.core_nz : (k + 1) * gp.core_nz,
      ] = v[
          i * gp.nx + hw : (i + 1) * gp.nx - hw,
          j * gp.ny + hw : (j + 1) * gp.ny - hw,
          k * gp.nz + hw : (k + 1) * gp.nz - hw,
      ]
    states_no_halos[key] = buf

  return states_no_halos


class TgvDriverSingleDeviceTest(absltest.TestCase):
  """Tests run_simulation end-to-end on a single device."""

  def _make_params(self):
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(_DRIVER_TGV_CONFIG_111, config)
    return parameters_lib.SwirlLMParameters(config)

  def test_run_simulation_completes(self):
    """Verifies run_simulation runs to completion on 1 device."""
    params = self._make_params()
    state = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=100,
        num_steps_per_cycle=3,
    )

    # Verify output has expected keys.
    for key in _STATE_KEYS:
      self.assertIn(key, state)

    # Verify no NaN/Inf.
    for key in _STATE_KEYS:
      self.assertTrue(
          jnp.all(jnp.isfinite(state[key])),
          f'Non-finite values in {key}',
      )

    state_no_halos = _remove_halos(params, state, ('u', 'v', 'w', 'p'))
    write_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')
    _ = visualization_lib.plot_overview(
        state_no_halos, save_path=f'{write_dir}/tgv_1x1x1.png'
    )

  def test_run_simulation_kinetic_energy_decays(self):
    """Verifies KE decays when run through the driver."""
    params = self._make_params()

    # Run 1 step to get reference KE.
    state_1 = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=1,
    )
    ke_1 = _kinetic_energy(state_1)

    # Run 5 steps.
    state_5 = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=5,
    )
    ke_5 = _kinetic_energy(state_5)

    self.assertGreater(ke_1, 0.0)
    self.assertLess(
        ke_5,
        ke_1,
        f'KE should decay: 1-step={ke_1}, 5-step={ke_5}',
    )

  def test_run_simulation_time_accumulates(self):
    """Verifies TIME_VARNAME is incremented correctly."""
    params = self._make_params()
    num_steps = 5
    state = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=num_steps,
    )

    expected_time = num_steps * params.dt
    actual_time = float(state[driver_lib.TIME_VARNAME])
    np.testing.assert_allclose(actual_time, expected_time, rtol=1e-6)


class TgvDriverMultiDeviceTest(absltest.TestCase):
  """Tests run_simulation end-to-end on 4 devices (1x1x4)."""

  def _make_params(self):
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(_DRIVER_TGV_CONFIG_221, config)
    return parameters_lib.SwirlLMParameters(config)

  def test_run_simulation_multi_device_completes(self):
    """Verifies run_simulation with 4 devices runs to completion."""
    params = self._make_params()
    state = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=100,
        num_steps_per_cycle=3,
    )

    # Verify output has expected keys.
    for key in _STATE_KEYS:
      self.assertIn(key, state)

    # Verify no NaN/Inf.
    for key in _STATE_KEYS:
      self.assertTrue(
          jnp.all(jnp.isfinite(state[key])),
          f'Non-finite values in {key}',
      )

    state_no_halos = _remove_halos(params, state, ('u', 'v', 'w', 'p'))
    write_dir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')
    _ = visualization_lib.plot_overview(
        state_no_halos, save_path=f'{write_dir}/tgv_2x2x1.png'
    )

  def test_run_simulation_multi_device_ke_decays(self):
    """Verifies KE decays on multi-device TGV."""
    params = self._make_params()

    state_1 = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=1,
    )
    state_no_halos_1 = _remove_halos(params, state_1, _STATE_KEYS)
    ke_1 = _kinetic_energy(state_no_halos_1, hw=0)

    state_5 = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=5,
    )
    state_no_halos_5 = _remove_halos(params, state_5, _STATE_KEYS)
    ke_5 = _kinetic_energy(state_no_halos_5, hw=0)

    self.assertGreater(ke_1, 0.0)
    self.assertLess(
        ke_5,
        ke_1,
        f'KE should decay: 1-step={ke_1}, 5-step={ke_5}',
    )

  def test_run_simulation_multi_device_time_accumulates(self):
    """Verifies TIME_VARNAME accumulates correctly on multi-device."""
    params = self._make_params()
    num_steps = 5
    state = driver_lib.run_simulation(
        params=params,
        init_fn=_tgv_driver_init_fn,
        num_steps=num_steps,
    )

    expected_time = num_steps * params.dt
    actual_time = float(state[driver_lib.TIME_VARNAME])
    np.testing.assert_allclose(actual_time, expected_time, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
