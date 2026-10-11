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
"""Tests for the signed distance functions.

Every primitive is checked against a distance worked out by hand rather than
against another implementation, so a shared sign or axis error cannot pass.
"""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.utility import signed_distance as sdf


def _pts(*rows) -> jnp.ndarray:
  return jnp.asarray(rows, dtype=jnp.float32)


class PrimitiveTest(parameterized.TestCase):
  """Distances computed by hand for points chosen to exercise each branch."""

  def test_sphere_distance(self):
    phi = sdf.sphere((1.0, 2.0, 3.0), 2.0)
    pts = _pts(
        (1.0, 2.0, 3.0),   # centre: -radius
        (4.0, 2.0, 3.0),   # 3 along +x: 3 - 2
        (1.0, 2.0, 5.0),   # on the surface
        (1.0, -2.0, 3.0),  # 4 along -y: 4 - 2
    )
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(phi, pts)),
        [-2.0, 1.0, 0.0, 2.0],
        atol=1e-6,
    )

  def test_sphere_normal_points_outward(self):
    phi = sdf.sphere((0.0, 0.0, 0.0), 1.0)
    pts = _pts((2.0, 0.0, 0.0), (0.0, -3.0, 0.0), (0.0, 0.0, 0.5))
    np.testing.assert_allclose(
        np.asarray(sdf.normal(phi, pts)),
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-6,
    )

  def test_box_face_edge_and_corner(self):
    # A unit cube centred at the origin: half extents 1 on every axis.
    phi = sdf.box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    pts = _pts(
        (0.0, 0.0, 0.0),    # centre: nearest face is 1 away, inside
        (3.0, 0.0, 0.0),    # face region: 3 - 1
        (2.0, 2.0, 0.0),    # edge region: sqrt(1 + 1)
        (2.0, 2.0, 2.0),    # corner region: sqrt(3)
        (0.5, 0.0, 0.0),    # inside, nearest face 0.5 away
    )
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(phi, pts)),
        [-1.0, 2.0, np.sqrt(2.0), np.sqrt(3.0), -0.5],
        atol=1e-6,
    )

  def test_box_rotated_90_degrees_matches_swapped_extents(self):
    # Rotating a 1x2x3 box by 90 degrees about z must equal a 2x1x3 box.
    rotated = sdf.box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), rotation_deg=(0, 0, 90))
    swapped = sdf.box((0.0, 0.0, 0.0), (2.0, 1.0, 3.0))
    rng = np.random.default_rng(0)
    pts = jnp.asarray(rng.uniform(-5.0, 5.0, size=(64, 3)), jnp.float32)
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(rotated, pts)),
        np.asarray(sdf.evaluate(swapped, pts)),
        atol=1e-5,
    )

  def test_infinite_cylinder_is_radial_distance(self):
    # Axis along z through (1, 1, *), radius 2. Distance is independent of z.
    phi = sdf.cylinder((1.0, 1.0, 0.0), (0.0, 0.0, 1.0), 2.0)
    pts = _pts(
        (1.0, 1.0, 0.0),    # on the axis
        (1.0, 1.0, 99.0),   # on the axis, far along it
        (6.0, 1.0, -5.0),   # 5 away radially
        (1.0, 3.0, 7.0),    # on the surface
    )
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(phi, pts)), [-2.0, -2.0, 3.0, 0.0], atol=1e-6
    )

  def test_cylinder_about_a_diagonal_axis(self):
    # The axis is the (1,1,0) diagonal: a case the height-field
    # representation could not express at all.
    phi = sdf.cylinder((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), 1.0)
    # (0,0,5) is 5 from the axis along z, so the distance is 5 - 1.
    # (1,1,0) lies on the axis, so it is -1.
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(phi, _pts((0.0, 0.0, 5.0), (1.0, 1.0, 0.0)))),
        [4.0, -1.0],
        atol=1e-6,
    )

  def test_finite_cylinder_caps(self):
    # Base at the origin, axis +z, radius 1, height 2.
    phi = sdf.cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, height=2.0)
    pts = _pts(
        (0.0, 0.0, 1.0),    # centre: 1 from the wall and 1 from each cap
        (0.0, 0.0, 5.0),    # 3 beyond the top cap
        (0.0, 0.0, -2.0),   # 2 below the base cap
        (3.0, 0.0, 1.0),    # 2 outside the wall
        (1.0 + 3.0, 0.0, 2.0 + 4.0),  # past the rim: sqrt(9 + 16)
    )
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(phi, pts)), [-1.0, 3.0, 2.0, 2.0, 5.0], atol=1e-6
    )

  def test_plane_is_signed_offset(self):
    phi = sdf.plane((0.0, 0.0, 2.0), (0.0, 0.0, 3.0))  # unnormalized normal
    np.testing.assert_allclose(
        np.asarray(
            sdf.evaluate(phi, _pts((9.0, -9.0, 5.0), (0.0, 0.0, 2.0),
                                   (0.0, 0.0, 0.0)))
        ),
        [3.0, 0.0, -2.0],
        atol=1e-6,
    )

  def test_height_field_is_the_vertical_gap(self):
    phi = sdf.height_field(lambda x, y: 0.5 * x + 0.0 * y)
    # At x = 2 the terrain is at z = 1, so z = 4 gives a gap of 3.
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(phi, _pts((2.0, 0.0, 4.0), (2.0, 0.0, 1.0)))),
        [3.0, 0.0],
        atol=1e-6,
    )


class ExactnessTest(parameterized.TestCase):
  """|grad phi| == 1 is the defining property of an exact distance function."""

  @parameterized.named_parameters(
      ('sphere', sdf.sphere((0.3, -0.2, 0.1), 1.0)),
      ('box', sdf.box((0.0, 0.1, 0.0), (0.7, 1.1, 0.4))),
      ('cylinder_z', sdf.cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.8)),
      ('cylinder_diag', sdf.cylinder((0.1, 0.0, 0.0), (1.0, 2.0, 3.0), 0.6)),
      ('plane', sdf.plane((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
  )
  def test_gradient_is_unit(self, phi):
    rng = np.random.default_rng(1)
    pts = jnp.asarray(rng.uniform(-4.0, 4.0, size=(256, 3)), jnp.float32)
    g = jax.vmap(jax.grad(phi))(pts)
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(g, axis=-1)), 1.0, atol=1e-4
    )

  def test_height_field_gradient_is_not_unit_on_a_slope(self):
    # Guards the documented caveat: `z - h` is a level set, not a distance.
    # On a 45-degree slope the gradient norm is sqrt(2), so a single Newton
    # step would land short -- which is why `project_to_surface` iterates.
    phi = sdf.height_field(lambda x, y: x + 0.0 * y)
    g = jax.vmap(jax.grad(phi))(_pts((1.0, 0.0, 5.0)))
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(g, axis=-1)), [np.sqrt(2.0)], atol=1e-6
    )


class CompositionTest(absltest.TestCase):

  def test_union_is_the_min(self):
    a = sdf.sphere((-2.0, 0.0, 0.0), 1.0)
    b = sdf.sphere((2.0, 0.0, 0.0), 1.0)
    both = sdf.union(a, b)
    pts = _pts((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    # Inside either sphere, and outside both at the midpoint (2 - 1 = 1).
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(both, pts)), [-1.0, -1.0, 1.0], atol=1e-6
    )

  def test_difference_carves_a_hole(self):
    solid = sdf.box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    hole = sdf.sphere((0.0, 0.0, 0.0), 1.0)
    cut = sdf.difference(solid, hole)
    # The box centre is now fluid, because the sphere was removed.
    self.assertGreater(float(sdf.evaluate(cut, _pts((0.0, 0.0, 0.0)))[0]), 0.0)
    # A point in the box but outside the sphere is still solid.
    self.assertLess(float(sdf.evaluate(cut, _pts((1.5, 0.0, 0.0)))[0]), 0.0)

  def test_intersection_keeps_only_the_overlap(self):
    a = sdf.sphere((-0.5, 0.0, 0.0), 1.0)
    b = sdf.sphere((0.5, 0.0, 0.0), 1.0)
    lens = sdf.intersection(a, b)
    # The origin is in both, so solid; (-1.2, 0, 0) is in `a` only, so fluid.
    self.assertLess(float(sdf.evaluate(lens, _pts((0.0, 0.0, 0.0)))[0]), 0.0)
    self.assertGreater(float(sdf.evaluate(lens, _pts((-1.2, 0.0, 0.0)))[0]), 0.0)

  def test_translate_matches_a_moved_primitive(self):
    moved = sdf.translate(sdf.sphere((0.0, 0.0, 0.0), 1.0), (1.0, 2.0, 3.0))
    direct = sdf.sphere((1.0, 2.0, 3.0), 1.0)
    rng = np.random.default_rng(2)
    pts = jnp.asarray(rng.uniform(-5.0, 5.0, size=(64, 3)), jnp.float32)
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(moved, pts)),
        np.asarray(sdf.evaluate(direct, pts)),
        atol=1e-5,
    )

  def test_rotate_leaves_a_sphere_unchanged(self):
    base = sdf.sphere((0.0, 0.0, 0.0), 1.5)
    spun = sdf.rotate(base, (30.0, 45.0, 60.0))
    rng = np.random.default_rng(3)
    pts = jnp.asarray(rng.uniform(-4.0, 4.0, size=(64, 3)), jnp.float32)
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(spun, pts)),
        np.asarray(sdf.evaluate(base, pts)),
        atol=1e-5,
    )


class ProjectionTest(parameterized.TestCase):
  """The body-intercept calculation of Ghias et al. (2007) section 2.5."""

  @parameterized.named_parameters(
      ('sphere', sdf.sphere((0.2, -0.1, 0.4), 1.3)),
      ('box', sdf.box((0.0, 0.0, 0.0), (1.0, 0.6, 1.4))),
      ('cylinder', sdf.cylinder((0.0, 0.0, 0.0), (1.0, 1.0, 2.0), 0.9)),
      ('union', sdf.union(sdf.sphere((-1.5, 0.0, 0.0), 1.0),
                          sdf.box((1.5, 0.0, 0.0), (0.8, 0.8, 0.8)))),
  )
  def test_projected_points_lie_on_the_surface(self, phi):
    rng = np.random.default_rng(4)
    pts = jnp.asarray(rng.uniform(-3.0, 3.0, size=(128, 3)), jnp.float32)
    bi = sdf.project_to_surface(phi, pts)
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(phi, bi)), 0.0, atol=1e-4
    )

  def test_offset_to_the_intercept_is_along_the_normal(self):
    # The defining property of a normal projection: GC - BI is parallel to n.
    phi = sdf.sphere((0.0, 0.0, 0.0), 1.0)
    rng = np.random.default_rng(5)
    pts = jnp.asarray(rng.uniform(-3.0, 3.0, size=(64, 3)), jnp.float32)
    bi = sdf.project_to_surface(phi, pts)
    n = sdf.normal(phi, pts)
    offset = pts - bi
    length = jnp.linalg.norm(offset, axis=-1, keepdims=True)
    # Skip points that happen to start on the surface.
    keep = np.asarray(length[:, 0]) > 1e-3
    cos = np.abs(np.asarray(jnp.sum(offset / length * n, axis=-1)))[keep]
    np.testing.assert_allclose(cos, 1.0, atol=1e-4)

  def test_one_step_suffices_for_an_exact_distance_function(self):
    phi = sdf.sphere((0.0, 0.0, 0.0), 1.0)
    rng = np.random.default_rng(6)
    pts = jnp.asarray(rng.uniform(-3.0, 3.0, size=(64, 3)), jnp.float32)
    one = sdf.project_to_surface(phi, pts, iterations=1)
    np.testing.assert_allclose(np.asarray(sdf.evaluate(phi, one)), 0.0, atol=1e-5)

  def test_curved_terrain_needs_more_than_one_step(self):
    # Newton is exact in one step for any *linear* level set, so a planar
    # slope converges immediately however steep it is. It is curvature that
    # costs iterations: on a Gaussian bump one step lands short of the
    # surface and three reach it. This is why the default is 3.
    phi = sdf.height_field(
        lambda x, y: 2.0 * jnp.exp(-(x**2 + y**2))
    )
    pts = _pts((0.7, 0.0, 2.2))
    one = sdf.evaluate(phi, sdf.project_to_surface(phi, pts, iterations=1))
    three = sdf.evaluate(phi, sdf.project_to_surface(phi, pts, iterations=3))
    self.assertGreater(float(np.abs(one[0])), 1e-3)
    np.testing.assert_allclose(np.asarray(three), 0.0, atol=1e-5)

  def test_planar_terrain_converges_in_one_step(self):
    # The companion to the test above: linear level set, one step is exact.
    phi = sdf.height_field(lambda x, y: x + 0.0 * y)
    one = sdf.evaluate(
        phi, sdf.project_to_surface(phi, _pts((0.0, 0.0, 4.0)), iterations=1)
    )
    np.testing.assert_allclose(np.asarray(one), 0.0, atol=1e-6)

  def test_image_point_is_the_reflection_through_the_intercept(self):
    # IP = 2*BI - GC, so the intercept is exactly midway (Ghias p. 538).
    phi = sdf.sphere((0.0, 0.0, 0.0), 1.0)
    gc = _pts((0.6, 0.0, 0.0), (0.0, 0.0, 0.9))
    bi = sdf.project_to_surface(phi, gc)
    ip = 2.0 * bi - gc
    midpoint = 0.5 * (gc + ip)
    np.testing.assert_allclose(np.asarray(midpoint), np.asarray(bi), atol=1e-6)
    # And the image point is in the fluid, as the construction requires.
    self.assertTrue(bool(jnp.all(sdf.evaluate(phi, ip) > 0.0)))


class GridTest(absltest.TestCase):

  def test_grid_points_shape_and_ordering(self):
    pts = sdf.grid_points(np.array([0.0, 1.0]), np.array([2.0, 3.0, 4.0]),
                          np.array([5.0]))
    self.assertEqual(pts.shape, (2, 3, 1, 3))
    np.testing.assert_allclose(np.asarray(pts[1, 2, 0]), [1.0, 4.0, 5.0])

  def test_mask_from_a_cylinder_on_a_grid(self):
    # A z-aligned cylinder of radius 1 at the centre of a 2x2 box.
    phi = sdf.cylinder((1.0, 1.0, 0.0), (0.0, 0.0, 1.0), 0.5)
    ax = np.linspace(0.05, 1.95, 20, dtype=np.float32)
    pts = sdf.grid_points(ax, ax, np.array([0.0], np.float32))
    solid = np.asarray(sdf.evaluate(phi, pts)) < 0.0
    # The solid region is a disc, so its area fraction is pi r^2 / 4.
    self.assertAlmostEqual(
        solid.mean(), np.pi * 0.5**2 / 4.0, delta=0.02
    )


if __name__ == '__main__':
  absltest.main()


class BatchedPrimitiveTest(parameterized.TestCase):
  """Many bodies of one type in a single primitive.

  Correctness is pinned by agreeing with the single-body constructors, which
  are themselves checked against distances worked out by hand above. The
  reason the batched form exists is compile time: one callable per body makes
  the traced graph grow with the body count.
  """

  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(11)
    self.pts = jnp.asarray(rng.uniform(-4.0, 4.0, size=(256, 3)), jnp.float32)

  def _agree(self, batched, singles):
    want = sdf.evaluate(sdf.union(*singles), self.pts)
    got = sdf.evaluate(batched, self.pts)
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(want), atol=1e-5
    )

  def test_spheres_match_the_single_body_union(self):
    centers = [(0.0, 0.0, 0.0), (2.0, -1.0, 0.5), (-3.0, 1.0, 1.0)]
    radii = [1.0, 0.5, 1.25]
    self._agree(
        sdf.batched_spheres(centers, radii),
        [sdf.sphere(c, r) for c, r in zip(centers, radii)],
    )

  def test_boxes_match_with_and_without_rotation(self):
    centers = [(0.0, 0.0, 0.0), (2.5, 1.0, 0.0)]
    halves = [(1.0, 0.7, 0.4), (0.5, 0.5, 0.5)]
    self._agree(
        sdf.batched_boxes(centers, halves),
        [sdf.box(c, h) for c, h in zip(centers, halves)],
    )
    rots = [(0.0, 0.0, 30.0), (15.0, 45.0, 0.0)]
    self._agree(
        sdf.batched_boxes(centers, halves, rots),
        [sdf.box(c, h, rotation_deg=r)
         for c, h, r in zip(centers, halves, rots)],
    )

  def test_cylinders_match_including_mixed_finite_and_infinite(self):
    bases = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    axes = [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
    radii = [0.8, 0.5]
    self._agree(
        sdf.batched_cylinders(bases, axes, radii),
        [sdf.cylinder(b, a, r) for b, a, r in zip(bases, axes, radii)],
    )
    # One capped, one infinite, in the same batch.
    self._agree(
        sdf.batched_cylinders(bases, axes, radii, [2.0, np.inf]),
        [sdf.cylinder(bases[0], axes[0], radii[0], height=2.0),
         sdf.cylinder(bases[1], axes[1], radii[1])],
    )

  @parameterized.named_parameters(
      ('spheres', lambda: sdf.batched_spheres(
          [[0, 0, 0], [2.5, 0, 0]], [1.0, 0.6])),
      ('boxes', lambda: sdf.batched_boxes(
          [[0, 0, 0], [3, 1, 0]], [[1, 0.7, 0.4], [0.5, 0.5, 0.5]])),
      ('boxes_rotated', lambda: sdf.batched_boxes(
          [[0, 0, 0]], [[1, 0.7, 0.4]], [[10.0, 20.0, 30.0]])),
      ('cylinders', lambda: sdf.batched_cylinders(
          [[0, 0, 0], [3, 0, 0]], [[0, 0, 1], [1, 1, 1]], [0.8, 0.5])),
      ('cylinders_capped', lambda: sdf.batched_cylinders(
          [[0, 0, 0]], [[0, 0, 1]], [0.8], [2.0])),
  )
  def test_gradient_is_unit_and_never_nan(self, make):
    # The single-body version had a NaN gradient throughout every box
    # interior, so the batched norms get the same guard and the same check.
    phi = make()
    g = jax.vmap(jax.grad(phi))(self.pts)
    self.assertFalse(bool(jnp.any(jnp.isnan(g))), 'gradient produced NaN')
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(g, axis=-1)), 1.0, atol=1e-4
    )

  def test_gradient_is_finite_at_interior_singularities(self):
    # Points where the batched norms see an exactly zero vector.
    for phi, pt in (
        (sdf.batched_boxes([[0, 0, 0]], [[1, 1, 1]]), (0.1, 0.0, 0.0)),
        (sdf.batched_spheres([[0, 0, 0]], [1.0]), (0.0, 0.0, 0.0)),
        (sdf.batched_cylinders([[0, 0, 0]], [[0, 0, 1]], [1.0]),
         (0.0, 0.0, 0.5)),
    ):
      g = jax.grad(phi)(jnp.asarray(pt, jnp.float32))
      self.assertTrue(bool(jnp.all(jnp.isfinite(g))), f'NaN gradient at {pt}')

  def test_many_bodies_stay_correct(self):
    rng = np.random.default_rng(5)
    centers = rng.uniform(-3.0, 3.0, size=(120, 3))
    radii = rng.uniform(0.1, 0.3, size=120)
    self._agree(
        sdf.batched_spheres(centers, radii),
        [sdf.sphere(tuple(c), float(r)) for c, r in zip(centers, radii)],
    )

  @parameterized.named_parameters(
      ('empty', lambda: sdf.batched_spheres(np.zeros((0, 3)), []), 'empty'),
      ('bad_width', lambda: sdf.batched_spheres([[0, 0]], [1.0]), 'shape'),
      ('negative_radius',
       lambda: sdf.batched_spheres([[0, 0, 0]], [-1.0]), 'positive'),
      ('mismatched_count',
       lambda: sdf.batched_spheres([[0, 0, 0]], [1.0, 2.0]), 'shape'),
      ('box_extent_mismatch',
       lambda: sdf.batched_boxes([[0, 0, 0]], [[1, 1, 1], [1, 1, 1]]),
       'match'),
      ('zero_axis',
       lambda: sdf.batched_cylinders([[0, 0, 0]], [[0, 0, 0]], [1.0]),
       'non-zero'),
  )
  def test_invalid_input_is_rejected(self, make, message):
    with self.assertRaisesRegex(ValueError, message):
      make()
