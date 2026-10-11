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
"""Tests for the cylinder validation case.

The diagnostics get the most attention here. `recirculation_length` and
`strouhal_number` are the instruments the validation is read off, so they are
checked against synthetic fields whose answers are known exactly; an
instrument that is wrong by a factor would make a correct solver look broken,
or worse, the reverse.
"""

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import jax
import numpy as np

from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.boundary_condition import immersed_boundary_ghost_point as gp
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.core import simulation as simulation_lib
from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.examples.cylinder import cylinder
from swirl_lm.jax.examples.cylinder import cylinder_main
from swirl_lm.jax.examples.cylinder import cylinder_pb2
from swirl_lm.jax.utility import global_layout
from swirl_lm.jax.utility import signed_distance as sdf


def _small_case(**overrides) -> cylinder_pb2.CylinderCase:
  """A case small enough to run in a test."""
  case = cylinder_pb2.CylinderCase(
      reynolds_number=40.0,
      length_upstream=3.0,
      length_downstream=6.0,
      half_width=3.0,
      span=0.5,
      cells_per_diameter=8,
      spanwise_cells=4,
      cfl=0.3,
  )
  for key, value in overrides.items():
    setattr(case, key, value)
  return case


class ConfigurationTest(parameterized.TestCase):
  """The case expands into a solver configuration that agrees with itself."""

  @parameterized.named_parameters(('re_40', 40.0), ('re_100', 100.0),
                                  ('re_3900', 3900.0))
  def test_viscosity_follows_the_reynolds_number(self, re):
    # The whole reason the case is a proto rather than a hand-written solver
    # config: these two cannot drift apart.
    sim = cylinder.Cylinder(_small_case(reynolds_number=re))
    self.assertAlmostEqual(
        sim.u_inf * sim.diameter / sim.params.nu, re, delta=re * 1e-4
    )

  def test_grid_and_time_step_follow_the_resolution(self):
    sim = cylinder.Cylinder(_small_case(cells_per_diameter=16))
    # 9 diameters streamwise at 16 points each.
    self.assertEqual(sim.fx, 9 * 16)
    self.assertEqual(sim.fy, 6 * 16)
    self.assertAlmostEqual(sim.dx, 9.0 / (9 * 16))
    self.assertAlmostEqual(sim.dt, 0.3 * sim.dx / sim.u_inf)

  def test_per_core_grid_includes_halos(self):
    # 8 spanwise cells over 2 cores is 4 each, so the per-core extent is 8 --
    # which the kernel widths divide. 4 cells over 2 cores would give 6 and
    # is rejected, see KernelWidthTest.
    sim = cylinder.Cylinder(_small_case(cores_z=2, spanwise_cells=8))
    grid = sim.params.grid_params
    self.assertEqual(grid.core_nz, 4)
    self.assertEqual(grid.nz, 4 + 2 * sim.case.halo_width)

  def test_the_span_is_the_periodic_direction(self):
    # The cylinder is uniform along the span, which is what makes a thin
    # periodic slab equivalent to the two-dimensional flow.
    sim = cylinder.Cylinder(_small_case())
    self.assertEqual(
        tuple(sim.params.grid_params.periodic_dims), (False, False, True)
    )

  def test_generated_config_round_trips(self):
    sim = cylinder.Cylinder(_small_case())
    self.assertIn('ghost_point', sim.config_text())
    # Re-parsing the dumped text must give the same configuration.
    from swirl_lm.base import parameters_pb2  # pylint: disable=g-import-not-at-top
    reparsed = parameters_pb2.SwirlLMParameters()
    text_format.Parse(sim.config_text(), reparsed)
    self.assertAlmostEqual(reparsed.kinematic_viscosity, sim.nu, places=6)

  @parameterized.named_parameters(
      ('negative_reynolds', {'reynolds_number': -1.0}, 'reynolds_number'),
      ('too_coarse', {'cells_per_diameter': 4}, 'not a circle'),
      ('thin_halo', {'halo_width': 1}, 'halo_width'),
  )
  def test_bad_cases_are_rejected(self, overrides, message):
    with self.assertRaisesRegex(ValueError, message):
      cylinder.Cylinder(_small_case(**overrides))

  def test_uneven_split_is_rejected_with_a_useful_message(self):
    # 4 spanwise cells across 3 cores does not divide.
    with self.assertRaisesRegex(ValueError, 'does not divide evenly'):
      cylinder.Cylinder(_small_case(spanwise_cells=4, cores_z=3))


class FarFieldTest(absltest.TestCase):
  """The boundaries, which got this case wrong the first time.

  The first Re = 100 run sheds correctly but the wake leaves the centreline
  and exits through the side of the domain. The cause was in this
  configuration, not in the immersed boundary: both lateral faces were zero
  gradient, which `find_bc_type.is_outflow` reads as an outlet, so nothing
  constrained transverse momentum; and with no pressure outlet the Poisson
  problem was singular. Each check below fails against that configuration.
  """

  def _bc_types(self, sim):
    return sim.params.bc_type

  def test_lateral_faces_are_not_outlets(self):
    sim = cylinder.Cylinder(_small_case())
    outflow = boundary_condition_utils.BoundaryType.OUTFLOW
    for face in (0, 1):
      self.assertNotEqual(
          self._bc_types(sim)[1][face], outflow,
          'a lateral face is classified OUTFLOW, so the wake can leave '
          'sideways and nothing holds it on the centreline',
      )

  def test_lateral_faces_carry_the_free_stream(self):
    # Ghias et al. section 2.4: "at the transverse boundaries we specify the
    # free stream velocity".
    sim = cylinder.Cylinder(_small_case())
    dirichlet = halo_exchange.BCType.DIRICHLET
    for face in (0, 1):
      self.assertEqual(sim.params.bc['u'][1][face][0], dirichlet)
      self.assertAlmostEqual(sim.params.bc['u'][1][face][1], sim.u_inf)
      self.assertEqual(sim.params.bc['v'][1][face][0], dirichlet)
      self.assertAlmostEqual(sim.params.bc['v'][1][face][1], 0.0)

  def test_the_outlet_is_still_an_outlet(self):
    sim = cylinder.Cylinder(_small_case())
    self.assertEqual(
        self._bc_types(sim)[0][1],
        boundary_condition_utils.BoundaryType.OUTFLOW,
    )
    self.assertEqual(
        self._bc_types(sim)[0][0],
        boundary_condition_utils.BoundaryType.INFLOW,
    )

  def test_pressure_is_pinned_somewhere(self):
    # A homogeneous Neumann condition on every non-periodic face leaves the
    # Poisson problem singular: the level is undetermined and the compatible
    # net boundary flux is not enforced, so the unconverged residual acts as
    # a spurious global pressure gradient.
    sim = cylinder.Cylinder(_small_case())
    simulation = simulation_lib.Simulation(sim.params)
    bc_p = simulation.pressure._bc['p']  # pylint: disable=protected-access
    pinned = [
        bc_p[d][f]
        for d in range(3)
        for f in (0, 1)
        if bc_p[d][f] is not None
        and bc_p[d][f][0] == halo_exchange.BCType.DIRICHLET
    ]
    self.assertNotEmpty(
        pinned,
        'every non-periodic pressure face is Neumann, so the Poisson '
        'problem is singular; set pressure_outlet',
    )

  def test_pressure_outlet_is_set_when_there_is_an_outlet(self):
    sim = cylinder.Cylinder(_small_case())
    has_outflow = any(
        sim.params.bc_type[d][f]
        == boundary_condition_utils.BoundaryType.OUTFLOW
        for d in range(3)
        for f in (0, 1)
    )
    if has_outflow:
      self.assertTrue(sim.params.pressure.pressure_outlet)

  def test_the_density_filter_is_off_for_constant_density(self):
    # `drho` is identically zero here, so filtering it is pure waste and
    # makes XLA constant-fold the stencil over the whole grid. See MEMORY.md.
    sim = cylinder.Cylinder(_small_case())
    self.assertEqual(sim.params.pressure.num_d_rho_filter, 0)


class KernelWidthTest(parameterized.TestCase):
  """Every per-core extent must divide by both kernel widths in play."""

  def test_the_default_case_divides(self):
    sim = cylinder.Cylinder(_small_case())
    grid = sim.params.grid_params
    for extent in (grid.nx, grid.ny, grid.nz):
      self.assertEqual(extent % grid.kernel_size, 0)
      self.assertEqual(extent % cylinder._CONVECTION_KERNEL_SIZE, 0)  # pylint: disable=protected-access

  def test_a_non_dividing_kernel_size_is_rejected(self):
    # 7 divides none of the extents; the error has to name the problem
    # rather than let the run silently take the other code path.
    with self.assertRaisesRegex(ValueError, 'not divisible'):
      cylinder.Cylinder(_small_case(kernel_size=7))

  def test_a_non_dividing_resolution_is_rejected(self):
    # An odd number of cells per diameter makes the extent odd, which the
    # hard-coded width of 4 in convection.py cannot divide.
    with self.assertRaisesRegex(ValueError, 'not divisible'):
      cylinder.Cylinder(_small_case(cells_per_diameter=9))


class GeometryTest(absltest.TestCase):
  """The body the solver sees is the body the case describes."""

  def test_the_cylinder_is_where_the_case_puts_it(self):
    sim = cylinder.Cylinder(_small_case())
    phi = sim.geometry_fn
    centre = np.array([[sim.center[0], sim.center[1], 0.25]], np.float32)
    # Negative at the axis, by one radius.
    self.assertAlmostEqual(
        float(sdf.evaluate(phi, centre)[0]), -0.5 * sim.diameter, places=5
    )
    # Zero on the surface.
    surface = centre + np.array([[0.5 * sim.diameter, 0.0, 0.0]], np.float32)
    self.assertAlmostEqual(float(sdf.evaluate(phi, surface)[0]), 0.0, places=5)

  def test_the_body_is_invariant_along_the_span(self):
    # This is what makes a span split the cheap decomposition: the
    # ghost-point stencil never reaches across a span seam.
    sim = cylinder.Cylinder(_small_case())
    pts = np.array(
        [[sim.center[0] + 0.3, sim.center[1], z] for z in (0.0, 0.1, 0.4)],
        np.float32,
    )
    values = np.asarray(sdf.evaluate(sim.geometry_fn, pts))
    np.testing.assert_allclose(values, values[0], atol=1e-6)

  def test_solid_fraction_matches_the_circle_area(self):
    sim = cylinder.Cylinder(_small_case(cells_per_diameter=16))
    mesh = driver_lib.create_mesh(sim.params)
    geometry = sim.ghost_points(mesh)
    interior = geometry.per_replica[0].interior_mask
    hw = sim.case.halo_width
    core = interior[hw:-hw, hw:-hw, hw:-hw]
    solid = 1.0 - core.mean()
    expected = np.pi * (0.5 * sim.diameter) ** 2 / (sim.lx * sim.ly)
    self.assertAlmostEqual(solid, expected, delta=0.2 * expected)


class DiagnosticsTest(absltest.TestCase):
  """The instruments the validation is read off."""

  def _field_with_bubble(self, sim, bubble_length):
    """A streamwise velocity that reverses for `bubble_length` diameters."""
    grid = sim.params.grid_params
    shape = (grid.nx, grid.ny, grid.nz)
    u = np.ones(shape, np.float32)
    hw = sim.case.halo_width
    # Interior x coordinates, matching what `_interior` will return.
    x = (np.arange(sim.fx) + 0.5) * sim.dx
    rear = sim.center[0] + 0.5 * sim.diameter
    reversed_region = (x > rear) & (x < rear + bubble_length * sim.diameter)
    line = np.where(reversed_region, -0.5, 1.0).astype(np.float32)
    # Write it on the wake axis, for every spanwise plane.
    j = hw + grid.core_ny // 2
    u[hw:hw + sim.fx, j, :] = line[:, None]
    return u

  def test_recirculation_length_recovers_a_known_bubble(self):
    sim = cylinder.Cylinder(_small_case(cores_z=1))
    for want in (1.0, 2.2, 3.5):
      u = self._field_with_bubble(sim, want)
      got = sim.recirculation_length({'u': u})
      # Resolved to one cell, which is dx/d = 1/cells_per_diameter.
      self.assertAlmostEqual(
          got, want, delta=2.0 / sim.case.cells_per_diameter,
          msg=f'wanted L/d = {want}, measured {got}',
      )

  def test_recirculation_length_is_zero_without_reversal(self):
    sim = cylinder.Cylinder(_small_case())
    grid = sim.params.grid_params
    u = np.ones((grid.nx, grid.ny, grid.nz), np.float32)
    self.assertEqual(sim.recirculation_length({'u': u}), 0.0)

  def test_strouhal_recovers_a_known_frequency(self):
    sim = cylinder.Cylinder(_small_case())
    dt = 0.01
    # St = f*d/u, so a target St fixes the frequency.
    for want in (0.1, 0.165, 0.3):
      freq = want * sim.u_inf / sim.diameter
      t = np.arange(4096) * dt
      signal = 0.3 * np.sin(2.0 * np.pi * freq * t) + 0.05
      got = sim.strouhal_number(signal, dt=dt)
      self.assertAlmostEqual(
          got, want, delta=0.02, msg=f'wanted St = {want}, measured {got}'
      )

  def test_strouhal_ignores_a_constant_offset(self):
    # A mean offset must not be read as a zero-frequency peak.
    sim = cylinder.Cylinder(_small_case())
    t = np.arange(2048) * 0.01
    signal = np.sin(2.0 * np.pi * 0.165 * t) + 7.0
    self.assertAlmostEqual(sim.strouhal_number(signal, dt=0.01), 0.165,
                           delta=0.02)

  def test_strouhal_returns_zero_for_a_dead_signal(self):
    sim = cylinder.Cylinder(_small_case())
    self.assertEqual(sim.strouhal_number(np.zeros(1024), dt=0.01), 0.0)
    self.assertEqual(sim.strouhal_number(np.zeros(4), dt=0.01), 0.0)

  def test_probe_is_downstream_and_off_axis(self):
    sim = cylinder.Cylinder(_small_case())
    i, j, _ = sim.probe_index()
    self.assertGreater((i + 0.5) * sim.dx, sim.center[0])
    self.assertNotEqual(j, sim.fy // 2)


class RunTest(absltest.TestCase):
  """A short run, to show the case is wired to the solver correctly."""

  def _run(self, case, num_steps=6):
    sim = cylinder.Cylinder(case)
    state = driver_lib.run_simulation(
        params=sim.params,
        init_fn=sim.initialization,
        num_steps=num_steps,
        num_steps_per_cycle=3,
    )
    return sim, state

  def test_the_case_runs_and_imposes_no_slip(self):
    sim, state = self._run(_small_case())
    u = np.asarray(state['u'])
    self.assertTrue(np.all(np.isfinite(u)))
    ghost = np.asarray(state['ib_boundary']) > 0.5
    self.assertGreater(ghost.sum(), 0)
    # Dirichlet zero makes each ghost value the mirror of its image point, so
    # a positive free stream leaves every ghost cell negative.
    self.assertTrue(
        np.all(u[ghost] < 0.0),
        f'{(u[ghost] >= 0).sum()} of {ghost.sum()} ghost cells are not '
        'mirrored',
    )

  def test_helper_fields_are_emitted(self):
    _, state = self._run(_small_case())
    for key in ('ib_interior_mask', 'ib_boundary', 'ib_phi'):
      self.assertIn(key, state)

  def test_a_span_split_gives_the_same_answer(self):
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    from swirl_lm.jax.utility import global_layout  # pylint: disable=g-import-not-at-top

    # 8 spanwise cells either way, so the global grid matches and both
    # per-core extents stay divisible by the kernel widths.
    sim_one, one = self._run(_small_case(cores_z=1, spanwise_cells=8))
    sim_two, two = self._run(_small_case(cores_z=2, spanwise_cells=8))
    a = global_layout.strip_halos_global(np.asarray(one['u']), sim_one.gp)
    b = global_layout.strip_halos_global(np.asarray(two['u']), sim_two.gp)
    self.assertEqual(a.shape, b.shape)
    worst = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-30)
    self.assertLess(worst, 1e-6, f'span split changed the answer by {worst:.2e}')


if __name__ == '__main__':
  absltest.main()


class DecompositionTest(parameterized.TestCase):
  """How many devices the case uses, and which splits are allowed.

  `driver.create_mesh` takes `jax.devices()[:cx*cy*cz]`, so the decomposition
  in the case file is the only thing that decides how many devices run. A
  separate "number of devices" setting can only disagree with it: asking for
  four GPUs while leaving the decomposition at 1x1x1 runs on one and idles
  three, silently.
  """

  def test_the_mesh_matches_the_requested_decomposition(self):
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    sim = cylinder.Cylinder(_small_case(cores_x=2))
    mesh = driver_lib.create_mesh(sim.params)
    self.assertEqual(mesh.shape['x'], 2)
    self.assertEqual(mesh.devices.size, 2)

  def test_a_smaller_decomposition_leaves_devices_idle(self):
    # Not an error -- a single-device baseline on a multi-GPU node is a
    # reasonable thing to want -- but it must be visible.
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    sim = cylinder.Cylinder(_small_case(cores_x=1, cores_y=1, cores_z=1))
    with self.assertLogs(level='WARNING') as logs:
      driver_lib.create_mesh(sim.params)
    self.assertRegex('\n'.join(logs.output), 'idle')

  def test_too_few_devices_is_reported_before_the_run(self):
    case = _small_case(cores_x=jax.device_count() + 1)
    with self.assertRaisesRegex(ValueError, 'device'):
      cylinder_main._check_device_count(case)  # pylint: disable=protected-access

  def test_enough_devices_passes(self):
    cylinder_main._check_device_count(_small_case())  # pylint: disable=protected-access

  @parameterized.named_parameters(
      ('x_two', {'cores_x': 2}), ('y_two', {'cores_y': 2}),
      ('span_two', {'cores_z': 2, 'spanwise_cells': 8}),
  )
  def test_legal_splits_build(self, overrides):
    sim = cylinder.Cylinder(_small_case(**overrides))
    grid = sim.params.grid_params
    for extent in (grid.nx, grid.ny, grid.nz):
      self.assertEqual(extent % grid.kernel_size, 0)

  def test_splitting_the_span_too_thin_names_spanwise_cells(self):
    # 4 spanwise cells over 2 cores gives 6 per core, which the kernel width
    # does not divide. The message has to name the knob that fixes it.
    with self.assertRaisesRegex(ValueError, 'spanwise_cells'):
      cylinder.Cylinder(_small_case(cores_z=2, spanwise_cells=4))

  def test_the_split_does_not_degrade_the_immersed_boundary(self):
    # Measured: no decomposition of this case forces a shortened probe or a
    # nearest-fluid fallback, not even one whose seam cuts the cylinder in
    # half. If that ever stops holding, the accuracy near the body drops to
    # first order on the affected cells and this is where it shows up.
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    for overrides in ({'cores_x': 2}, {'cores_y': 2}):
      sim = cylinder.Cylinder(_small_case(**overrides))
      geometry = sim.ghost_points(driver_lib.create_mesh(sim.params))
      capped = sum(g.num_capped for g in geometry.per_replica)
      fallback = sum(g.num_fallback for g in geometry.per_replica)
      self.assertEqual(capped, 0, f'{overrides}: {capped} probes shortened')
      self.assertEqual(fallback, 0, f'{overrides}: {fallback} fallback cells')

  def test_the_span_needs_no_halo_for_this_body(self):
    # The cylinder is invariant along the span, so an image point never moves
    # in z. That is what makes the span the cheapest axis to split.
    sim = cylinder.Cylinder(_small_case())
    geometry = sim.ghost_points(driver_lib.create_mesh(sim.params))
    reach_z = max(g.max_reach_cells[2] for g in geometry.per_replica)
    self.assertAlmostEqual(reach_z, 0.0, places=5)


class SeamThroughTheBodyTest(absltest.TestCase):
  """A device seam cutting the cylinder in half.

  This is the configuration no earlier test covered, and it was wrong. The
  span split that `RunTest` exercises is trivial for the immersed boundary --
  the cylinder is invariant along the span, so no stencil crosses the seam --
  and the isolated `apply_local` tests hold the field fixed, so neither could
  see a stale halo. Splitting y puts the seam straight through the body, and
  the error showed up as a 7% band two cells wide along the seam, confined to
  the streamwise extent of the cylinder: the ghost-point update writes cells
  that a neighbour holds a copy of, and without a second halo exchange that
  copy keeps its pre-update value.
  """

  def _run(self, cores_x, cores_y, steps=8):
    case = _small_case(cores_x=cores_x, cores_y=cores_y)
    sim = cylinder.Cylinder(case)
    state = driver_lib.run_simulation(
        params=sim.params,
        init_fn=sim.initialization,
        num_steps=steps,
        num_steps_per_cycle=steps,
    )
    return sim, {
        key: global_layout.strip_halos_global(
            np.asarray(state[key]), sim.gp
        )
        for key in ('u', 'v', 'p')
    }

  def test_a_seam_through_the_body_changes_nothing(self):
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    _, reference = self._run(1, 1)
    # cores_y = 2 puts the seam on the wake centreline, which is the
    # cylinder's horizontal diameter.
    sim, split = self._run(1, 2)
    for key in ('u', 'v', 'p'):
      a, b = reference[key], split[key]
      scale = max(float(np.abs(a).max()), 1e-30)
      worst = float(np.abs(a - b).max()) / scale
      self.assertLess(
          worst, 1e-6,
          f'{key} depends on the partitioning: max relative difference '
          f'{worst:.3e}. The seam cuts the cylinder, so this is most likely '
          'a halo copy of a ghost cell going stale.',
      )

  def test_a_seam_clear_of_the_body_also_changes_nothing(self):
    # The control: with cores_x the seam falls downstream of the cylinder.
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    _, reference = self._run(1, 1)
    _, split = self._run(2, 1)
    for key in ('u', 'v', 'p'):
      a, b = reference[key], split[key]
      scale = max(float(np.abs(a).max()), 1e-30)
      self.assertLess(float(np.abs(a - b).max()) / scale, 1e-6, key)


class ProbeTest(absltest.TestCase):
  """The wake probe, which is how the shedding frequency is measured."""

  def test_the_mask_selects_exactly_one_global_cell(self):
    sim = cylinder.Cylinder(_small_case())
    mask = np.asarray(sim.probe_mask(sim.params, (0, 0, 0)))
    self.assertEqual(mask.sum(), 1.0)

  def test_the_probe_reads_the_cell_it_claims_to(self):
    sim = cylinder.Cylinder(_small_case())
    mesh = driver_lib.create_mesh(sim.params)
    record = []
    state = driver_lib.run_simulation(
        params=sim.params, init_fn=sim.initialization, num_steps=4,
        num_steps_per_cycle=4, probe_fn=sim.probe_fn(mesh),
        probe_sink=record.append,
    )
    samples = np.concatenate(record)
    self.assertLen(samples, 4)
    v = global_layout.strip_halos_global(np.asarray(state['v']), sim.gp)
    i, j, k = sim.probe_index()
    self.assertAlmostEqual(float(samples[-1]), float(v[i, j, k]), places=6)

  def test_the_probe_survives_a_split(self):
    # The reduction is a psum, so no replica needs to know whether the probe
    # is its own.
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    out = []
    for cores_x in (1, 2):
      sim = cylinder.Cylinder(_small_case(cores_x=cores_x))
      mesh = driver_lib.create_mesh(sim.params)
      record = []
      driver_lib.run_simulation(
          params=sim.params, init_fn=sim.initialization, num_steps=4,
          num_steps_per_cycle=4, probe_fn=sim.probe_fn(mesh),
          probe_sink=record.append,
      )
      out.append(np.concatenate(record))
    np.testing.assert_allclose(out[0], out[1], atol=1e-6)

  def test_no_probe_means_no_cost_and_no_change(self):
    # `probe_fn=None` must leave the existing behaviour untouched.
    sim = cylinder.Cylinder(_small_case())
    a = driver_lib.run_simulation(
        params=sim.params, init_fn=sim.initialization, num_steps=4,
        num_steps_per_cycle=4,
    )
    mesh = driver_lib.create_mesh(sim.params)
    b = driver_lib.run_simulation(
        params=sim.params, init_fn=sim.initialization, num_steps=4,
        num_steps_per_cycle=4, probe_fn=sim.probe_fn(mesh),
        probe_sink=lambda _: None,
    )
    np.testing.assert_array_equal(np.asarray(a['u']), np.asarray(b['u']))

  def test_strouhal_is_resolved_to_a_thousandth(self):
    # Parabolic interpolation of the spectral peak: without it the answer is
    # quantised to the bin width, which for a realistic record is the same
    # size as the difference between the values being compared against.
    sim = cylinder.Cylinder(_small_case())
    rng = np.random.default_rng(0)
    for want in (0.1573, 0.1650, 0.1734):
      step = 20 * sim.dt
      n = 650
      t = np.arange(n) * step
      signal = np.sin(2.0 * np.pi * want * t) + 0.02 * rng.standard_normal(n)
      got = sim.strouhal_number(signal, dt=step)
      self.assertAlmostEqual(got, want, delta=0.003)
