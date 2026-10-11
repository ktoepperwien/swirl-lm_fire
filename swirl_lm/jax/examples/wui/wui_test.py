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
"""Tests for the wildland-urban-interface case.

Weighted towards the things that actually went wrong while this was built,
all of which were silent:

  * a **frame disagreement**. The boxes are written in one rotation
    convention and read in another, and nothing complains if they disagree --
    the houses simply end up somewhere else. The overlay drew them wrong for
    a whole afternoon for exactly this reason, so the convention is pinned
    here against `batched_boxes` itself rather than against a copy of the
    formula.
  * a **surface lying on the cell centres**. The elevation model is whole
    metres and the vertical spacing is one metre, which is the degenerate
    case of the ghost-point construction. It cost 18% of the terrain's ghost
    cells before it was found, and it announces itself only as a number in a
    log line.
  * **an unconstrained solid interior**, which is a quarter of this domain.

The scene file is synthesised rather than read, so these run without the
Mandeville data.
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

from swirl_lm.jax.boundary_condition import immersed_boundary_ghost_point as gp
from swirl_lm.jax.boundary_condition import immersed_boundary_method
from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.examples.wui import wui
from swirl_lm.jax.examples.wui import wui_pb2
from swirl_lm.jax.utility import signed_distance as sdf


def _write_scene(
    path: str,
    shape=(16, 16, 24),
    spacing=(2.0, 2.0, 1.0),
    terrain=None,
    boxes=1,
    fuel_height=3,
) -> str:
  """Writes a small scene file in the layout `extract_scene.py` produces.

  Args:
    path: Where to write it.
    shape: `(nx, ny, nz)` cells.
    spacing: `(dx, dy, dz)` in metres.
    terrain: Optional `(nx, ny)` terrain height above the floor. Defaults to
      a ramp, which is the interesting case: flat ground hides every
      orientation and interpolation error.
    boxes: How many building boxes to place.
    fuel_height: Cells of vegetation above the ground.

  Returns:
    `path`.
  """
  nx, ny, nz = shape
  dx, dy, dz = spacing
  if terrain is None:
    ramp = np.linspace(2.0, 8.0, nx, dtype=np.float32)
    terrain = np.repeat(ramp[:, None], ny, axis=1)
  terrain = np.asarray(terrain, np.float32)

  centres, halves, rotations = [], [], []
  for n in range(boxes):
    centres.append([10.0 + 6.0 * n, 12.0, 8.0])
    halves.append([3.0, 2.0, 5.0])
    rotations.append([0.0, 0.0, 20.0 + 10.0 * n])

  fuel = np.zeros((nx, ny, nz), np.float32)
  z = (np.arange(nz) + 0.5) * dz
  above = z[None, None, :] - terrain[:, :, None]
  fuel[(above >= 0.0) & (above < fuel_height * dz)] = 0.4

  np.savez_compressed(
      path,
      box_centres=np.asarray(centres, np.float32).reshape(-1, 3),
      box_half_extents=np.asarray(halves, np.float32).reshape(-1, 3),
      box_rotations_deg=np.asarray(rotations, np.float32).reshape(-1, 3),
      terrain_height=terrain,
      fuel_bulk_density=fuel,
      origin_utm=np.asarray([388760.0, 3779400.0, 295.0], np.float64),
      spacing=np.asarray(spacing, np.float64),
      shape=np.asarray(shape, np.int64),
      metadata=np.asarray('{"buildings": []}'),
  )
  return path


def _case(path: str, **overrides) -> wui_pb2.WuiCase:
  """A case small enough to run in a test."""
  case = wui_pb2.WuiCase(scene_path=path, pressure_iterations=4)
  for key, value in overrides.items():
    setattr(case, key, value)
  return case


class SceneTest(parameterized.TestCase):
  """The scene file decides the mesh, and disagreements are caught."""

  def test_mesh_comes_from_the_scene_at_the_stated_spacing(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))

    # The nodes span `[0, length]` inclusive, so the length is one interval
    # short of the cell count. Getting this wrong stretches every cell by
    # `n / (n - 1)`, which is how `dz` quietly stops being the 1 m the
    # terrain was checked at.
    self.assertAlmostEqual(sim.gp.dx, 2.0, places=6)
    self.assertAlmostEqual(sim.gp.dy, 2.0, places=6)
    self.assertAlmostEqual(sim.gp.dz, 1.0, places=6)
    self.assertEqual((sim.fx, sim.fy, sim.fz), (16, 16, 24))

  def test_rejects_a_scene_whose_arrays_disagree(self):
    directory = self.create_tempdir()
    path = os.path.join(directory, 'scene.npz')
    _write_scene(path)
    data = dict(np.load(path))
    data['terrain_height'] = data['terrain_height'][:-1]
    np.savez_compressed(path, **data)
    with self.assertRaisesRegex(ValueError, 'terrain_height'):
      wui.Wui(_case(path))

  def test_rejects_a_vertical_spacing_the_terrain_is_not_safe_at(self):
    path = _write_scene(
        os.path.join(self.create_tempdir(), 'scene.npz'),
        spacing=(2.0, 2.0, 2.0),
    )
    with self.assertRaisesRegex(ValueError, 'dz = 2'):
      wui.Wui(_case(path))

  def test_requires_a_scene_path(self):
    with self.assertRaisesRegex(ValueError, 'scene_path'):
      wui.Wui(wui_pb2.WuiCase())


class GeometryFrameTest(parameterized.TestCase):
  """The boxes land where the scene file says they do."""

  def test_box_rotation_matches_the_signed_distance_convention(self):
    """A point on the box's own axis is inside it.

    This is the test that would have caught the overlay bug. A box at yaw
    `theta` has its local `+x` axis along `(cos theta, sin theta)` in the
    world, so a point half a half-extent along that direction from the
    centre must be inside. Under the opposite sign convention, for a long
    thin box, it is outside.
    """
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    centre = sim.box_centres[0]
    half = sim.box_half_extents[0]
    yaw = np.radians(sim.box_rotations[0, 2])

    along = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    inside = centre + 0.9 * half[0] * along
    outside = centre + 1.8 * half[0] * along
    boxes = sdf.batched_boxes(
        sim.box_centres, sim.box_half_extents, sim.box_rotations
    )
    self.assertLess(float(boxes(jnp.asarray(inside, jnp.float32))), 0.0)
    self.assertGreater(float(boxes(jnp.asarray(outside, jnp.float32))), 0.0)

    # ... and across the short axis the same distances come out the other
    # way round, which is what makes the previous check sensitive to the
    # sign rather than to the box being merely large.
    across = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    self.assertGreater(
        float(boxes(jnp.asarray(centre + 1.8 * half[0] * across,
                                jnp.float32))),
        0.0,
    )

  def test_terrain_profile_interpolates_the_table_at_cell_centres(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    profile = sim._raw_terrain_profile()  # pylint: disable=protected-access
    for index in (0, 5, 15):
      x = (index + 0.5) * sim.dx
      self.assertAlmostEqual(
          float(profile(x, 7.0)), float(sim.terrain_height[index, 3]), places=4
      )

  def test_terrain_profile_holds_the_edge_value_outside_the_table(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    profile = sim._raw_terrain_profile()  # pylint: disable=protected-access
    edge = float(profile(0.5 * sim.dx, 0.5 * sim.dy))
    self.assertAlmostEqual(float(profile(-50.0, -50.0)), edge, places=5)

  def test_terrain_profile_is_differentiable(self):
    """`signed_distance.normal` needs this; a table lookup alone would not."""
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    profile = sim._raw_terrain_profile()  # pylint: disable=protected-access
    slope = float(jax.grad(lambda x: profile(x, 7.0))(11.0))
    # The synthetic terrain ramps from 2 to 8 m over 15 intervals of 2 m.
    self.assertAlmostEqual(slope, 6.0 / 30.0, places=4)


class NodeClearanceTest(parameterized.TestCase):
  """A surface through the cell centres is the degenerate case; find it."""

  def test_whole_metre_terrain_on_a_one_metre_grid_is_shifted(self):
    path = _write_scene(
        os.path.join(self.create_tempdir(), 'scene.npz'),
        terrain=np.full((16, 16), 5.0, np.float32),
    )
    sim = wui.Wui(_case(path))
    self.assertGreater(sim.terrain_shift, 0.0)

    # Every node column now clears the surface.
    z = np.asarray(sim.gp.global_xyz_with_halos[2])
    gap = np.abs(z - (5.0 + sim.terrain_shift))
    self.assertGreater(gap.min(), 0.05 * sim.dz)

  def test_terrain_already_clear_of_the_centres_is_left_alone(self):
    path = _write_scene(
        os.path.join(self.create_tempdir(), 'scene.npz'),
        terrain=np.full((16, 16), 5.4, np.float32),
    )
    self.assertEqual(wui.Wui(_case(path)).terrain_shift, 0.0)

  def test_the_shift_moves_the_initial_condition_too(self):
    """The geometry and the wind have to see the same ground."""
    path = _write_scene(
        os.path.join(self.create_tempdir(), 'scene.npz'),
        terrain=np.full((16, 16), 5.0, np.float32),
    )
    sim = wui.Wui(_case(path))
    profile = sim.terrain_profile()
    self.assertAlmostEqual(
        float(profile(10.0, 10.0)), 5.0 + sim.terrain_shift, places=5
    )

  def test_a_flat_scene_has_no_capped_probes_or_fallback_cells(self):
    """The end the shift exists for, measured rather than assumed."""
    path = _write_scene(
        os.path.join(self.create_tempdir(), 'scene.npz'),
        terrain=np.full((16, 16), 5.0, np.float32),
        boxes=0,
    )
    sim = wui.Wui(_case(path))
    geometry = sim.ghost_points(driver_lib.create_mesh(sim.params))
    self.assertEqual(sim.fallback_fraction(geometry), 0.0)


class DecompositionTest(parameterized.TestCase):
  """The checks that keep a split from being silently wrong."""

  def test_rejects_splitting_the_vertical(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    with self.assertRaisesRegex(ValueError, 'cores_z'):
      wui.Wui(_case(path, cores_z=2))

  def test_rejects_a_halo_narrower_than_the_convection_scheme_needs(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    with self.assertRaisesRegex(ValueError, 'halo_width'):
      wui.Wui(_case(path, halo_width=1))

  def test_rejects_a_mesh_that_does_not_divide_across_the_cores(self):
    path = _write_scene(
        os.path.join(self.create_tempdir(), 'scene.npz'), shape=(12, 16, 24)
    )
    with self.assertRaisesRegex(ValueError, 'does not divide evenly'):
      wui.Wui(_case(path, cores_x=8))

  def test_rejects_a_per_core_extent_the_kernel_width_does_not_divide(self):
    # 16 points over 2 cores is 8 each, plus 2*3 halo cells is 14, which 4
    # does not divide -- the silent slice-fallback case.
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    with self.assertRaisesRegex(ValueError, 'not divisible by 4'):
      wui.Wui(_case(path, cores_x=2, halo_width=3))


class InflowTest(parameterized.TestCase):
  """The log law, and the inlet planes derived from it."""

  def test_profile_passes_through_the_reference_point(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path, inflow_speed=5.0, reference_height=10.0))
    self.assertAlmostEqual(float(sim.inflow_profile(10.0)), 5.0, places=4)

  def test_profile_vanishes_at_the_ground_and_below_it(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    self.assertAlmostEqual(float(sim.inflow_profile(0.0)), 0.0, places=6)
    self.assertAlmostEqual(float(sim.inflow_profile(-20.0)), 0.0, places=6)

  def test_time_step_follows_the_fastest_wind_not_the_reference_speed(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path, cfl=0.3))
    self.assertGreater(sim.u_max, sim.u_ref)
    self.assertAlmostEqual(
        sim.dt, 0.3 * min(sim.dx, sim.dy, sim.dz) / sim.u_max, places=8
    )

  def test_initialization_supplies_the_inlet_planes(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    states = sim.initialization(sim.params, (0, 0, 0))
    halo = sim.gp.halo_width
    for name in ('u', 'v', 'w'):
      key = f'bc_{name}_0_0'
      self.assertIn(key, states)
      self.assertEqual(states[key].shape[0], halo + 1)
      # The plane is the field's own inlet slice, so it cannot drift from it.
      np.testing.assert_allclose(
          np.asarray(states[key]), np.asarray(states[name])[: halo + 1]
      )

  def test_the_wind_follows_the_terrain_rather_than_the_floor(self):
    """On a ramp, a fixed height above the floor sees different speeds."""
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    states = sim.initialization(sim.params, (0, 0, 0))
    u = np.asarray(states['u'])
    mask = np.asarray(states[gp.INTERIOR_MASK_KEY])
    halo = sim.gp.halo_width
    level = halo + 12            # 12 m above the floor, inside the fluid
    west = u[halo + 1, halo + 8, level]
    east = u[halo + 14, halo + 8, level]
    self.assertTrue(mask[halo + 1, halo + 8, level] > 0.5)
    self.assertTrue(mask[halo + 14, halo + 8, level] > 0.5)
    # The ground rises eastward, so the same floor-relative level is closer
    # to the ground there and the wind is slower.
    self.assertLess(east, west)


class VegetationTest(parameterized.TestCase):
  """The fuel field, and the drag it produces."""

  def test_fuel_reaches_the_state_where_the_scene_puts_it(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    states = sim.initialization(sim.params, (0, 0, 0))
    halo = sim.gp.halo_width
    interior = np.asarray(states['rho_f'])[
        halo:-halo, halo:-halo, halo:-halo
    ]
    np.testing.assert_allclose(interior, sim.fuel, atol=1e-6)

  def test_drag_opposes_the_flow_and_scales_with_the_fuel(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path, c_d=0.2, a_v_per_density=1.0))
    library = sim.source_term_update_fn()

    states = {
        'rho': jnp.ones((2, 1, 1)),
        'u': jnp.array([[[3.0]], [[3.0]]]),
        'v': jnp.zeros((2, 1, 1)),
        'w': jnp.zeros((2, 1, 1)),
    }
    additional = {'rho_f': jnp.array([[[1.0]], [[2.0]]])}
    src = np.asarray(library['u'](states, additional)['src_u'])

    self.assertLess(src[0, 0, 0], 0.0)
    # Twice the fuel, twice the drag: the whole point of making `a_v` follow
    # the density rather than holding it uniform as the fire example does.
    self.assertAlmostEqual(src[1, 0, 0], 2.0 * src[0, 0, 0], places=5)
    expected = -3.0 / 8.0 * 0.2 * 1.0 * 1.0 * 3.0 * 3.0
    self.assertAlmostEqual(src[0, 0, 0], expected, places=5)

  def test_no_drag_where_there_is_no_fuel(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    library = sim.source_term_update_fn()
    states = {
        'rho': jnp.ones((1, 1, 1)),
        'u': jnp.array([[[4.0]]]),
        'v': jnp.zeros((1, 1, 1)),
        'w': jnp.zeros((1, 1, 1)),
    }
    src = library['u'](states, {'rho_f': jnp.zeros((1, 1, 1))})['src_u']
    self.assertEqual(float(src[0, 0, 0]), 0.0)


class ConfigurationTest(parameterized.TestCase):
  """The generated solver configuration says what the case means."""

  def test_every_body_reaches_the_geometry(self):
    path = _write_scene(
        os.path.join(self.create_tempdir(), 'scene.npz'), boxes=3
    )
    sim = wui.Wui(_case(path))
    geometry = sim.params.boundary_models.ib.ghost_point.geometry
    kinds = [message.WhichOneof('shape') for message in geometry]
    self.assertEqual(kinds.count('box'), 3)
    self.assertEqual(kinds.count('height_field'), 1)

  def test_the_lateral_faces_are_not_classified_as_outlets(self):
    """All three components Neumann on a face makes it an outlet, and with
    `pressure_outlet` that pins the pressure along the side of the domain.
    On the cylinder it sent the wake out through the side wall."""
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    bc = sim.params.bc
    for dim, face in ((1, 0), (1, 1), (2, 1)):
      kinds = [str(bc[name][dim][face][0]) for name in ('u', 'v', 'w')]
      self.assertTrue(
          any(kind.endswith('DIRICHLET') for kind in kinds),
          msg=f'face (dim {dim}, location {face}) is entirely zero '
              f'gradient, which `is_outflow` reads as an outlet: {kinds}',
      )

  def test_the_solid_interior_is_not_clamped(self):
    """Holding the solid at the boundary value destabilises the fluid.

    The solver integrates momentum in solid cells too, so a held cell feeds
    its neighbour on the next step and the step in the field marches out
    until the five-point diffusion stencil reads it. Measured on flat
    ground: clamped, non-finite by step 200; unclamped, the inflow value
    held exactly. Left alone the solid stays smooth.
    """
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    self.assertFalse(
        sim.params.boundary_models.ib.ghost_point.clamp_interior
    )

  def test_the_configuration_round_trips(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    self.assertIn('clamp_interior: false', sim.config_text())
    self.assertIn('num_d_rho_filter: 0', sim.config_text())


class InteriorClampTest(parameterized.TestCase):
  """The solid below the ghost layer is held, not integrated."""

  def _geometry_states(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    geometry = sim.ghost_points(driver_lib.create_mesh(sim.params))
    states = {
        key: jnp.asarray(value)
        for key, value in geometry.initial_states((0, 0, 0)).items()
    }
    # `from_additional_states` runs inside `shard_map`, where every array
    # still carries the three replica axes.
    local = gp.from_additional_states({
        key: value[None, None, None] if key.startswith(gp.GEOMETRY_PREFIX)
        else value
        for key, value in states.items()
    })
    return sim, states, local

  def test_deep_interior_mask_is_the_solid_minus_the_ghost_layer(self):
    _, states, local = self._geometry_states()
    mask = np.asarray(
        gp.deep_interior_mask(states[gp.INTERIOR_MASK_KEY], local)
    )
    solid = np.asarray(states[gp.INTERIOR_MASK_KEY]) < 0.5
    ghost = np.asarray(states[gp.GHOST_MASK_KEY]) > 0.5

    self.assertTrue(np.all(mask <= solid), 'marked a fluid cell')
    self.assertFalse(np.any(mask & ghost), 'marked a cell the method sets')
    # And nothing within reach of the fluid's own stencils, which is the
    # whole point: a held cell inside that reach is a step in the field the
    # five-point diffusion stencil can see. Dilated with scipy rather than
    # `np.roll`, which would wrap the far face onto the near one.
    near = ndimage.binary_dilation(
        np.asarray(states[gp.INTERIOR_MASK_KEY]) > 0.5,
        ndimage.generate_binary_structure(3, 1),
        iterations=gp._STENCIL_REACH,  # pylint: disable=protected-access
    )
    self.assertFalse(np.any(mask & near), 'marked a cell the fluid reads')
    # Not vacuous: most of the solid lies below the ghost layer, which is
    # the whole reason this matters on a scene like this one.
    self.assertGreater(mask.sum(), ghost.sum())

  def test_update_states_holds_the_deep_interior_at_the_boundary_value(self):
    sim, states, local = self._geometry_states()
    # Switched on for this test only; the case deliberately leaves it off.
    sim.params.boundary_models.ib.ghost_point.clamp_interior = True
    ib = immersed_boundary_method.ImmersedBoundaryMethod(sim.params)
    field = jnp.ones_like(states[gp.INTERIOR_MASK_KEY])
    updated = np.asarray(
        ib.update_states({'u': field}, dict(states))['u']
    )

    held = np.asarray(
        gp.deep_interior_mask(states[gp.INTERIOR_MASK_KEY], local)
    )
    self.assertTrue(held.any(), 'nothing was deep enough to hold')
    np.testing.assert_allclose(updated[held], 0.0, atol=1e-6)
    # The fluid is left alone; only the unreachable part of the body is.
    solid = np.asarray(states[gp.INTERIOR_MASK_KEY]) < 0.5
    np.testing.assert_allclose(updated[~solid], 1.0, atol=1e-6)

  def test_the_clamp_is_off_unless_asked_for(self):
    sim, states, _ = self._geometry_states()
    ib = immersed_boundary_method.ImmersedBoundaryMethod(sim.params)
    field = jnp.ones_like(states[gp.INTERIOR_MASK_KEY])
    updated = np.asarray(ib.update_states({'u': field}, dict(states))['u'])

    solid = np.asarray(states[gp.INTERIOR_MASK_KEY]) < 0.5
    ghost = np.asarray(states[gp.GHOST_MASK_KEY]) > 0.5
    deep = solid & ~ghost
    np.testing.assert_allclose(updated[deep], 1.0, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
