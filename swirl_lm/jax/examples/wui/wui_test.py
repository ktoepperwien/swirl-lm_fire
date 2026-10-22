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
from google.protobuf import text_format
from scipy import ndimage

from swirl_lm.base import parameters_pb2

from swirl_lm.jax.boundary_condition import immersed_boundary_ghost_point as gp
from swirl_lm.jax.boundary_condition import boundary_condition_utils
from swirl_lm.jax.boundary_condition import immersed_boundary_method
from swirl_lm.jax.communication import halo_exchange_utils
from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.equations import pressure as pressure_lib
from swirl_lm.jax.examples.wui import extract_scene
from swirl_lm.jax.examples.wui import wui
from swirl_lm.jax.examples.wui import scene_to_vtk
from swirl_lm.jax.examples.wui import wui_pb2
from swirl_lm.jax.physics.thermodynamics import manager as thermodynamics_manager
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


class TargetShapeTest(parameterized.TestCase):
  """An exact cell count, for a solver that constrains its extents.

  Multigrid needs a power of two per core, which the crop will never happen to
  be. The padding absorbs the difference -- terrain extended at its edge
  value, fuel zero -- so the extra is clear fluid and the only question is
  whether enough of it lands on each face.
  """

  def _scene(self, nx=328, ny=216, nz=4):
    terrain = np.zeros((nx, ny), np.float32)
    fuel = np.zeros((nx, ny, nz), np.float32)
    boxes = {'box_centres': np.array([[10.0, 20.0, 5.0]], np.float32)}
    return terrain, fuel, boxes, (0.0, 0.0, float(nx), float(ny))

  def test_the_mesh_reaches_the_target_exactly(self):
    terrain, fuel, boxes, crop = self._scene()

    padded, fuel_padded, _, _ = extract_scene.pad_with_margin(
        terrain, fuel, boxes, crop, 1.0, 50.0, target=(512, 512)
    )

    self.assertEqual(padded.shape, (512, 512))
    self.assertEqual(fuel_padded.shape[:2], (512, 512))

  def test_boxes_shift_by_the_low_side_padding_only(self):
    # The low-side pad is what moves the origin, so that is what the centres
    # move by. Using half the total would be wrong whenever the split is odd.
    terrain, fuel, boxes, crop = self._scene()

    _, _, shifted, _ = extract_scene.pad_with_margin(
        terrain, fuel, boxes, crop, 1.0, 50.0, target=(512, 512)
    )

    np.testing.assert_allclose(
        shifted['box_centres'][0, :2],
        boxes['box_centres'][0, :2] + np.array([92.0, 148.0]),
    )

  def test_an_odd_difference_puts_the_extra_cell_on_the_high_side(self):
    self.assertEqual(extract_scene._target_padding(328, 513, 56, 'x'), (92, 93))

  def test_a_target_that_starves_a_face_is_refused(self):
    # The failure this guards is silent under-padding: a box within `dx` of a
    # face puts the face's condition and the immersed boundary's on the same
    # cells, and the run goes non-finite within ten steps.
    with self.assertRaisesRegex(ValueError, 'clear fluid'):
      extract_scene._target_padding(328, 336, 56, 'x')

  def test_a_target_below_the_crop_is_refused(self):
    with self.assertRaisesRegex(ValueError, 'cannot remove cells'):
      extract_scene._target_padding(328, 300, 56, 'x')

  def test_without_a_target_the_margin_alone_decides(self):
    # The behaviour before the flag existed, which has to be preserved: a
    # symmetric pad of the margin rounded up to whole cell blocks.
    terrain, fuel, boxes, crop = self._scene()

    padded, _, _, _ = extract_scene.pad_with_margin(
        terrain, fuel, boxes, crop, 1.0, 50.0
    )

    self.assertEqual(padded.shape, (328 + 112, 216 + 112))


class SurfaceFuelTest(parameterized.TestCase):
  """The surface layer, which arrives as a load rather than a density.

  `tree/bulkDensity` is kg/m^3 already. `surface/fuelLoad` is kg/m^2 over a
  separate `surface/fuelDepth`, so the density it stands for is the load
  divided by that depth. Getting this wrong is silent: the drag reads `rho_f`
  directly and a load used as a density is wrong by a factor of the depth,
  which on this data is between two and eleven.
  """

  def _grid(self, load=0.25, depth=0.5, canopy=1.0, elevation=10.0):
    ny, nx, levels = 4, 4, 3
    return extract_scene.Grid(
        elevation=np.full((ny, nx), elevation, np.float64),
        bulk_density=np.full((levels, ny, nx), canopy, np.float32),
        surface_load=np.full((ny, nx), load, np.float64),
        surface_depth=np.full((ny, nx), depth, np.float64),
        west=0.0,
        south=0.0,
        north=float(ny),
        dx=1.0,
        dz=1.0,
    )

  def _resample(self, grid, nz=8, dz=1.0):
    terrain = np.full((4, 4), float(grid.elevation[0, 0]), np.float64)
    z_floor = float(terrain.min()) - 2.0
    fuel = extract_scene.resample_fuel(
        grid, terrain, (0.0, 0.0, 4.0, 4.0), 1.0, dz, z_floor, nz
    )
    hag = (z_floor + (np.arange(nz) + 0.5) * dz)[None, None, :] - terrain[
        :, :, None
    ]
    return fuel, hag

  def test_the_column_integral_is_the_load(self):
    # The invariant: the drag on a column is proportional to `sum(rho_f) * dz`,
    # so the load is what must survive, not the in-layer density.
    grid = self._grid(load=0.25, depth=0.5, canopy=0.0)

    fuel, _ = self._resample(grid)

    np.testing.assert_allclose(fuel.sum(axis=2) * 1.0, 0.25, atol=1e-6)

  def test_a_layer_thinner_than_a_cell_lands_in_the_first_cell(self):
    # At 1 m every surface layer on this data is thinner than a cell, and the
    # first cell centre above ground sits 0.5 m up while the layer is at most
    # 0.46 m deep -- so testing the centre against the layer drops the fuel
    # entirely. That was the first version of this code.
    grid = self._grid(load=0.25, depth=0.4, canopy=0.0)

    fuel, hag = self._resample(grid)

    first = np.argmax(hag[0, 0] > 0.0)
    self.assertAlmostEqual(float(fuel[0, 0, first]), 0.25, places=6)
    np.testing.assert_allclose(
        np.delete(fuel[0, 0], first), 0.0, atol=1e-6
    )

  def test_a_cell_inside_a_thick_layer_gets_the_in_layer_density(self):
    # The other limit. A layer several cells deep gives each interior cell the
    # density within it, `load / depth`.
    grid = self._grid(load=1.0, depth=4.0, canopy=0.0)

    fuel, hag = self._resample(grid)

    deep = (hag > 0.5) & (hag < 3.5)
    np.testing.assert_allclose(fuel[deep], 1.0 / 4.0, atol=1e-6)

  def test_the_surface_and_the_canopy_add(self):
    grid = self._grid(load=0.25, depth=0.4, canopy=1.0)

    fuel, hag = self._resample(grid)

    # Above the surface layer, the canopy alone.
    above = (hag >= 1.0) & (hag < 3.0)
    np.testing.assert_allclose(fuel[above], 1.0, atol=1e-6)
    # And the first cell carries both.
    first = np.argmax(hag[0, 0] > 0.0)
    self.assertAlmostEqual(float(fuel[0, 0, first]), 1.0 + 0.25, places=6)

  def test_the_surface_stays_at_or_below_its_depth(self):
    grid = self._grid(load=0.25, depth=0.4, canopy=0.0)

    fuel, hag = self._resample(grid)

    np.testing.assert_allclose(fuel[hag < -0.5], 0.0, atol=1e-6)
    np.testing.assert_allclose(fuel[hag > 1.0], 0.0, atol=1e-6)

  def test_the_column_integral_does_not_depend_on_the_mesh(self):
    grid = self._grid(load=0.25, depth=0.4, canopy=0.0)

    coarse, _ = self._resample(grid, nz=8, dz=1.0)
    fine, _ = self._resample(grid, nz=32, dz=0.25)

    self.assertAlmostEqual(float(coarse.sum(axis=2).max() * 1.0), 0.25, places=6)
    self.assertAlmostEqual(float(fine.sum(axis=2).max() * 0.25), 0.25, places=6)

  def test_a_zero_depth_contributes_nothing_rather_than_an_infinity(self):
    # Zero depth is how the store says there is no surface fuel here. A naive
    # division would put an infinity into a field the drag reads directly.
    grid = self._grid(load=0.25, depth=0.0, canopy=0.0)

    fuel, _ = self._resample(grid)

    self.assertTrue(np.all(np.isfinite(fuel)))
    np.testing.assert_allclose(fuel, 0.0, atol=1e-6)


class RotationTest(parameterized.TestCase):
  """A half turn about the vertical axis, which is how an east wind is said.

  The solver blows along +x and nothing downstream takes a wind direction, so
  the scene is turned rather than the wind.
  """

  def _scene(self, nx=12, ny=8, nz=5, dx=1.0):
    rng = np.random.default_rng(0)
    terrain = rng.standard_normal((nx, ny)).astype(np.float32)
    fuel = rng.standard_normal((nx, ny, nz)).astype(np.float32)
    boxes = {
        'box_centres': np.array(
            [[2.0, 3.0, 4.0], [9.0, 1.0, 2.0]], np.float32
        ),
        'box_half_extents': np.array(
            [[1.0, 2.0, 3.0], [2.0, 1.0, 1.0]], np.float32
        ),
        'box_rotations_deg': np.array(
            [[0.0, 0.0, 35.0], [0.0, 0.0, 68.5]], np.float32
        ),
    }
    return terrain, fuel, boxes, dx

  def test_both_horizontal_axes_reverse(self):
    # Reversing only x would be a reflection, not a rotation: the footprints
    # would come out mirrored, so a wall facing the wind on one side of a
    # ridge would face away from it on the other.
    terrain, fuel, boxes, dx = self._scene()

    rotated_terrain, rotated_fuel, _ = extract_scene.rotate_half_turn(
        terrain, fuel, boxes, dx
    )

    np.testing.assert_array_equal(rotated_terrain, terrain[::-1, ::-1])
    np.testing.assert_array_equal(rotated_fuel, fuel[::-1, ::-1, :])

  def test_box_centres_reflect_about_the_node_inclusive_span(self):
    # `Wui` derives `lx = (nx - 1) * dx`, so the centre maps to `lx - x` and
    # not to `nx * dx - x`. The difference is half a cell on every building.
    terrain, fuel, boxes, dx = self._scene()
    lx = (terrain.shape[0] - 1) * dx
    ly = (terrain.shape[1] - 1) * dx

    _, _, rotated = extract_scene.rotate_half_turn(terrain, fuel, boxes, dx)

    np.testing.assert_allclose(
        rotated['box_centres'][:, 0], lx - boxes['box_centres'][:, 0]
    )
    np.testing.assert_allclose(
        rotated['box_centres'][:, 1], ly - boxes['box_centres'][:, 1]
    )

  def test_vertical_placement_and_shape_are_untouched(self):
    # A half turn is horizontal. The yaw is left alone because a box is
    # symmetric under it: `yaw + 180` describes the same body.
    terrain, fuel, boxes, dx = self._scene()

    _, _, rotated = extract_scene.rotate_half_turn(terrain, fuel, boxes, dx)

    np.testing.assert_array_equal(
        rotated['box_centres'][:, 2], boxes['box_centres'][:, 2]
    )
    np.testing.assert_array_equal(
        rotated['box_half_extents'], boxes['box_half_extents']
    )
    np.testing.assert_array_equal(
        rotated['box_rotations_deg'], boxes['box_rotations_deg']
    )

  def test_turning_twice_is_the_identity(self):
    terrain, fuel, boxes, dx = self._scene()
    before = np.array(boxes['box_centres'], copy=True)

    once = extract_scene.rotate_half_turn(terrain, fuel, boxes, dx)
    twice = extract_scene.rotate_half_turn(*once, dx)

    np.testing.assert_array_equal(twice[0], terrain)
    np.testing.assert_array_equal(twice[1], fuel)
    np.testing.assert_allclose(twice[2]['box_centres'], before)
    # And the caller's arrays were not written through.
    np.testing.assert_array_equal(boxes['box_centres'], before)


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
    sim = wui.Wui(_case(path, c_d=0.2, a_v_per_density=1.0,
                        sponge_fraction=0.0))
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
    sim = wui.Wui(_case(path, sponge_fraction=0.0))
    library = sim.source_term_update_fn()
    states = {
        'rho': jnp.ones((1, 1, 1)),
        'u': jnp.array([[[4.0]]]),
        'v': jnp.zeros((1, 1, 1)),
        'w': jnp.zeros((1, 1, 1)),
    }
    src = library['u'](states, {'rho_f': jnp.zeros((1, 1, 1))})['src_u']
    self.assertEqual(float(src[0, 0, 0]), 0.0)


class SpongeTest(parameterized.TestCase):
  """Rayleigh damping at the outlet, and on the sides in place of periodicity."""

  def test_the_outlet_is_damped_and_the_sides_optionally(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    for lateral, expected in ((False, 1), (True, 3)):
      sim = wui.Wui(_case(path, lateral_sponge_fraction=(0.08 if lateral else 0.0)))
      layers = list(sim.params.sponge)
      self.assertLen(layers, 1)
      self.assertLen(layers[0].orientation, expected)
      # The outlet is always one of them: high face of x.
      self.assertIn((0, 1), [(o.dim, o.face) for o in layers[0].orientation])

  def test_zero_fraction_switches_it_off_entirely(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path, sponge_fraction=0.0))
    self.assertIsNone(sim.sponge)
    self.assertNotIn('sponge_beta', sim.initialization(sim.params, (0, 0, 0)))

  def test_the_profile_is_zero_inside_and_rises_towards_the_outlet(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path, sponge_fraction=0.2, sponge_coeff=5.0,
                        lateral_sponge_fraction=0.0))
    states = sim.initialization(sim.params, (0, 0, 0))
    beta = np.asarray(states['sponge_beta'])
    halo = sim.gp.halo_width
    self.assertAlmostEqual(float(beta.max()), 1.0 / 5.0, places=4)
    # Undamped well upstream, damped at the outlet.
    self.assertAlmostEqual(float(beta[halo + 1, :, :].max()), 0.0, places=6)
    self.assertGreater(float(beta[-halo - 1, :, :].max()), 0.0)

  def test_the_targets_are_the_undisturbed_inlet(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    states = sim.initialization(sim.params, (0, 0, 0))
    for name in ('u', 'v', 'w'):
      np.testing.assert_allclose(
          np.asarray(states[f'{name}_init']), np.asarray(states[name]))

  def test_the_force_opposes_a_departure_from_the_target(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path, sponge_fraction=0.2, lateral_sponge_fraction=0.0))
    library = sim.source_term_update_fn()
    states = sim.initialization(sim.params, (0, 0, 0))
    perturbed = dict(states)
    # Push the whole field above its target; the damping must pull back.
    perturbed['u'] = states['u'] + 5.0
    src = np.asarray(library['u'](perturbed, states)['src_u'])
    beta = np.asarray(states['sponge_beta'])
    self.assertLess(src[beta > 0].min(), 0.0)
    # And leave the interior alone, where there is no fuel either.
    quiet = (beta == 0) & (np.asarray(states['rho_f']) == 0)
    np.testing.assert_allclose(src[quiet], 0.0, atol=1e-6)


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
    On the cylinder it sent the wake out through the side wall.

    The lid is deliberately excluded -- see the test below. The sides are what
    the cylinder failed on and they stay walls.
    """
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    bc = sim.params.bc
    for dim, face in ((1, 0), (1, 1)):
      kinds = [str(bc[name][dim][face][0]) for name in ('u', 'v', 'w')]
      self.assertTrue(
          any(kind.endswith('DIRICHLET') for kind in kinds),
          msg=f'face (dim {dim}, location {face}) is entirely zero '
              f'gradient, which `is_outflow` reads as an outlet: {kinds}',
      )

  def test_a_closed_lid_is_not_an_outlet(self):
    # The alternative, for when the top is far enough above the scene that
    # the plume never reaches it. Only `w` differs between the two.
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path, open_top=False))
    bc = sim.params.bc

    self.assertIn('DIRICHLET', str(bc['w'][2][1][0]))
    self.assertNotEqual(
        boundary_condition_utils.BoundaryType.OUTFLOW,
        sim.params.bc_type[2][1],
    )
    # And then the outlet at x-high is the only pinned pressure face, which
    # is what every run before this one had.
    pressure = pressure_lib.Pressure(
        sim.params, thermodynamics_manager.ThermodynamicsManager(sim.params)
    )
    dirichlet = [
        (dim, face)
        for dim in range(3)
        for face in range(2)
        if pressure._bc['p'][dim][face] is not None  # pylint: disable=protected-access
        and pressure._bc['p'][dim][face][0]  # pylint: disable=protected-access
        == halo_exchange_utils.BCType.DIRICHLET
    ]
    self.assertCountEqual(dirichlet, [(0, 1)])

  def test_the_lid_is_an_outlet_so_the_plume_can_leave(self):
    """A rising plume held at the top by `w = 0` goes sideways instead.

    The recirculation that produces is a property of the box, not of the fire.
    This is the one face where being classified as an outlet is wanted, so it
    is asserted rather than merely allowed -- the previous configuration had
    `w` Dirichlet here and the assertion above covered it.
    """
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    bc = sim.params.bc

    kinds = [str(bc[name][2][1][0]) for name in ('u', 'v', 'w')]
    self.assertTrue(
        all('NEUMANN' in kind for kind in kinds),
        msg=f'the lid is not entirely zero gradient: {kinds}',
    )
    self.assertEqual(
        boundary_condition_utils.BoundaryType.OUTFLOW,
        sim.params.bc_type[2][1],
    )

  def test_the_pressure_is_pinned_at_both_outlets(self):
    """`pressure_outlet` makes every OUTFLOW face Dirichlet for `p`.

    With the lid open there are two, which keeps the Poisson problem
    non-singular as the single outlet did.
    """
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    sim = wui.Wui(_case(path))
    pressure = pressure_lib.Pressure(
        sim.params, thermodynamics_manager.ThermodynamicsManager(sim.params)
    )

    dirichlet = [
        (dim, face)
        for dim in range(3)
        for face in range(2)
        if pressure._bc['p'][dim][face] is not None  # pylint: disable=protected-access
        and pressure._bc['p'][dim][face][0]  # pylint: disable=protected-access
        == halo_exchange_utils.BCType.DIRICHLET
    ]

    self.assertCountEqual(dirichlet, [(0, 1), (2, 1)])

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


class SceneExportTest(parameterized.TestCase):
  """The ParaView export, whose two failure modes are both silent.

  A transposed array opens, has the right size, and shows the domain the wrong
  way round. A negated yaw draws every box but those at yaw zero somewhere
  else -- the mistake the fitting overlay made for an afternoon.
  """

  def test_the_box_corners_follow_the_signed_distance_convention(self):
    # The same check `GeometryFrameTest` makes of the solver's own boxes: a
    # point along the box's local +x axis is inside it. Here it is made of the
    # corners the exporter emits, so the two cannot drift apart.
    centre = np.array([10.0, 20.0, 5.0])
    half = np.array([6.0, 1.5, 3.0])
    yaw = 35.0

    corners = scene_to_vtk.box_corners(centre, half, yaw)

    angle = np.radians(yaw)
    along = np.array([np.cos(angle), np.sin(angle), 0.0])
    # The long axis is local x, so the corners must reach further along
    # `along` than across it.
    reach_along = np.abs((corners - centre) @ along).max()
    across = np.array([-np.sin(angle), np.cos(angle), 0.0])
    reach_across = np.abs((corners - centre) @ across).max()
    self.assertAlmostEqual(reach_along, 6.0, places=5)
    self.assertAlmostEqual(reach_across, 1.5, places=5)

  def test_the_corners_are_a_hexahedron_in_vtk_order(self):
    corners = scene_to_vtk.box_corners(
        np.zeros(3), np.array([1.0, 2.0, 3.0]), 0.0
    )

    self.assertEqual(corners.shape, (8, 3))
    # The first four share the base, the last four the top.
    np.testing.assert_allclose(corners[:4, 2], -3.0)
    np.testing.assert_allclose(corners[4:, 2], +3.0)
    # And the base is traversed rather than crossed: consecutive corners
    # differ in exactly one horizontal coordinate.
    for a, b in zip(corners[:4], np.roll(corners[:4], -1, axis=0)):
      self.assertEqual(int(np.sum(~np.isclose(a[:2], b[:2]))), 1)

  def test_the_vtk_ordering_puts_x_fastest(self):
    # The archive is indexed x first, which in C order makes z fastest --
    # exactly the opposite of what VTK reads.
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    flat = scene_to_vtk._to_vtk_order(array)  # pylint: disable=protected-access

    self.assertEqual(flat.shape, (24,))
    # The first two entries differ by one step in x, not in z.
    self.assertEqual(float(flat[0]), float(array[0, 0, 0]))
    self.assertEqual(float(flat[1]), float(array[1, 0, 0]))


class SolverChoiceTest(parameterized.TestCase):
  """Which Poisson solver the generated configuration selects.

  Jacobi remains the default: it is what every result from this case so far
  was produced with, and at `pressure_iterations` it leaves most of the
  divergence in place -- which is why no number from this case should be
  quoted. The other two exist so that can change.
  """

  @parameterized.named_parameters(
      ('jacobi', 'JACOBI', 'jacobi'),
      ('conjugate_gradient', 'CONJUGATE_GRADIENT', 'conjugate_gradient'),
      ('multigrid', 'MULTIGRID', 'multigrid'),
  )
  def test_the_generated_config_selects_the_requested_solver(
      self, name, expected
  ):
    # Parsed rather than string-matched, and the oneof read back: the text is
    # checked against the TensorFlow proto, which is where a field spelled for
    # the wrong one shows up.
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    case = _case(path)
    case.solver = getattr(wui_pb2.WuiCase, name)
    sim = wui.Wui(case)

    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(sim.config_text(), config)

    self.assertEqual(config.pressure.solver.WhichOneof('solver'), expected)

  def test_the_iteration_count_reaches_every_solver(self):
    # One knob, three meanings -- sweeps, iterations, V-cycles -- so each
    # branch has to read it and the useful values differ by orders of
    # magnitude.
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    for name, field in (
        ('JACOBI', 'max_iterations'),
        ('CONJUGATE_GRADIENT', 'max_iterations'),
        ('MULTIGRID', 'num_iterations'),
    ):
      case = _case(path, pressure_iterations=7)
      case.solver = getattr(wui_pb2.WuiCase, name)
      config = parameters_pb2.SwirlLMParameters()
      text_format.Parse(wui.Wui(case).config_text(), config)

      selected = getattr(
          config.pressure.solver, config.pressure.solver.WhichOneof('solver')
      )
      self.assertEqual(getattr(selected, field), 7, msg=name)

  def test_conjugate_gradient_asks_for_a_relative_tolerance(self):
    # `atol` is absolute unless `l2_norm_reduction` is set, and the norm of
    # the right hand side scales with the mesh and the flow -- so one fixed
    # number means a converged solve on this grid and an immediate stop on
    # another.
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    case = _case(path)
    case.solver = wui_pb2.WuiCase.CONJUGATE_GRADIENT

    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(wui.Wui(case).config_text(), config)

    self.assertTrue(config.pressure.solver.conjugate_gradient.l2_norm_reduction)
    # And `reprojection` is left for `Pressure` to settle from the boundary
    # conditions; with an outlet pinning the level it must come out off.
    self.assertFalse(
        config.pressure.solver.conjugate_gradient.HasField('reprojection')
    )

  def test_multigrid_stops_coarsening_at_eight(self):
    # Four cells cannot hold the five-point Galerkin coarse operator; on the
    # fire case that diverged to 1.8e+09 after eight cycles against 1.99e-04
    # stopping at eight.
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))
    case = _case(path)
    case.solver = wui_pb2.WuiCase.MULTIGRID

    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(wui.Wui(case).config_text(), config)

    shape = config.pressure.solver.multigrid.coarsest_subgrid_shape
    self.assertEqual((shape.dim_0, shape.dim_1, shape.dim_2), (8, 8, 8))

  def test_jacobi_is_the_default(self):
    path = _write_scene(os.path.join(self.create_tempdir(), 'scene.npz'))

    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(wui.Wui(_case(path)).config_text(), config)

    self.assertEqual(config.pressure.solver.WhichOneof('solver'), 'jacobi')


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


class SyntheticBuildingTest(parameterized.TestCase):
  """The single-building mode, which exists to isolate one clean box.

  The fitted geometry confounds the building with two other things: boxes
  too thin for the grid, and gaps between neighbours that are equally
  unresolved. One well-resolved block on flat ground has neither.
  """

  def _terrain(self, nx=60, ny=60, height=10.0):
    """Flat ground with a steep ramp along one edge, so the search has to
    actually prefer the flat part rather than landing anywhere."""
    terrain = np.full((nx, ny), height, np.float32)
    terrain[:, :8] += np.linspace(40.0, 0.0, 8)[None, :]
    return terrain

  @parameterized.parameters(3, 6, 8, 10)
  def test_the_box_is_the_requested_size_in_cells(self, cells):
    terrain = self._terrain()
    boxes = extract_scene.single_building(
        terrain, dx=2.0, dz=1.0, cells=cells, yaw_deg=45.0,
        position=(60.0, 60.0))
    half = boxes['box_half_extents'][0]
    self.assertLen(boxes['box_centres'], 1)
    self.assertAlmostEqual(2 * half[0] / 2.0, cells, places=5)
    self.assertAlmostEqual(2 * half[1] / 2.0, cells, places=5)
    # Vertical: the requested height plus the burial depth.
    visible = boxes['box_centres'][0, 2] + half[2] - 10.0
    self.assertAlmostEqual(visible, cells * 1.0, places=5)

  def test_the_box_sits_on_the_ground_rather_than_floating(self):
    terrain = self._terrain(height=10.0)
    boxes = extract_scene.single_building(
        terrain, 2.0, 1.0, 8, 45.0, position=(60.0, 60.0))
    centre, half = boxes['box_centres'][0], boxes['box_half_extents'][0]
    base = centre[2] - half[2]
    # Buried, so no gap can open between the block and the terrain.
    self.assertLess(base, 10.0)
    self.assertAlmostEqual(base, 10.0 - extract_scene._BURY_DEPTH_M, places=5)  # pylint: disable=protected-access

  def test_the_yaw_is_about_z_only(self):
    boxes = extract_scene.single_building(
        self._terrain(), 2.0, 1.0, 8, 45.0, position=(60.0, 60.0))
    np.testing.assert_allclose(
        boxes['box_rotations_deg'][0], [0.0, 0.0, 45.0], atol=1e-6)

  def test_a_box_with_no_interior_is_refused(self):
    """Under three cells every solid cell is on the surface, which is the
    defect the fitted geometry has."""
    with self.assertRaisesRegex(ValueError, 'under 3 cells'):
      extract_scene.single_building(
          self._terrain(), 2.0, 1.0, 2, 45.0, position=(60.0, 60.0))

  def test_the_placement_search_prefers_flat_ground(self):
    terrain = self._terrain()
    x, y = extract_scene.flattest_patch(terrain, dx=2.0, window_m=24.0)
    # The ramp occupies the first 8 columns, y < 16 m.
    self.assertGreater(y, 16.0)

  def test_no_buildings_is_an_empty_but_well_shaped_set(self):
    boxes = extract_scene._no_buildings()  # pylint: disable=protected-access
    for key in ('box_centres', 'box_half_extents', 'box_rotations_deg'):
      self.assertEqual(boxes[key].shape, (0, 3))

  def test_resolvability_report_runs_on_an_empty_scene(self):
    extract_scene.report_resolvability(
        extract_scene._no_buildings(), 2.0, 1.0)  # pylint: disable=protected-access


if __name__ == '__main__':
  absltest.main()
