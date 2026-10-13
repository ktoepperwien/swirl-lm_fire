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
"""Tests for the ghost-point immersed boundary geometry.

The central check is `ExactnessTest`: trilinear interpolation reproduces a
linear field exactly, and the normal extrapolation is itself linear, so for a
planar boundary and a linear field satisfying the boundary condition the ghost
value must come back to machine precision. That single property pins the sign
of `zeta`, the magnitude of `Gamma`, the corner ordering, the index
arithmetic and the probe-length algebra all at once, so it is run across
probe lengths, boundary-condition types, grid stretching and axis orderings.
"""

import glob
import logging as pylog
import os
import shutil

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import jax
from jax.experimental import mesh_utils
from jax.experimental import shard_map
import jax.numpy as jnp
import numpy as np
import zarr
from swirl_lm.base import parameters_pb2
from swirl_lm.boundary_condition import immersed_boundary_method_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.boundary_condition import immersed_boundary_ghost_point as gp
from swirl_lm.jax.utility import global_layout
from swirl_lm.jax.utility import grid_parametrization as gp_lib
from swirl_lm.jax.utility import signed_distance as sdf

# The geometry is stored in float32 and the sweeps accumulate, so "exact"
# means to float32 round-off. On fields of magnitude ~10 this is a relative
# 1e-6, which is far tighter than any sign, index or algebra error could
# survive -- those show up at O(1).
_EXACT = 1e-5


def _reliable(geom) -> np.ndarray:
  """Ghost cells whose value cannot have been polluted by a first-order one.

  Two treatments are deliberately first-order and so excluded from any
  exactness claim.

  A **fallback** cell uses the nearest-fluid approximation.

  A cell **anchored on the body intercept** has had the weight of stencil
  nodes nothing ever writes moved onto the intercept. The intercept carries
  the surface value, which for a linear field is the value at a different
  point, so the interpolation is no longer exact there. That is the trade the
  anchoring makes, and it is worth making: before it, those nodes were read
  straight out of the drifting interior of the body. Note this test harness
  leaves the true field in those cells, so the old exactness result was an
  artefact of the harness rather than a property of a real run.

  Either kind also appears as a stencil corner for its neighbours, and those
  neighbours for theirs, so the exclusion has to be taken to its transitive
  closure -- measured, it spreads exactly two levels for an oblique plane,
  and stopping at one level leaves a 2e-4 residual that looks like a bug but
  is not.
  """
  bad = (geom.fallback | (geom.bi_weight > 1e-6)).copy()
  while True:
    bad_flat = set(geom.gc_index[bad].tolist())
    grown = bad.copy()
    for row, (corners, weights) in enumerate(
        zip(geom.corner_index, geom.alpha)
    ):
      if any(
          int(c) in bad_flat and w > 1e-12 for c, w in zip(corners, weights)
      ):
        grown[row] = True
    if np.array_equal(grown, bad):
      return ~bad
    bad = grown


def _uniform(n: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
  h = (hi - lo) / n
  return np.linspace(lo + 0.5 * h, hi - 0.5 * h, n)


def _stretched(n: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
  """A smoothly graded axis, finer at the low end."""
  t = np.linspace(0.0, 1.0, n + 1) ** 1.6
  edges = lo + (hi - lo) * t
  return 0.5 * (edges[:-1] + edges[1:])


class GhostCellMaskTest(absltest.TestCase):

  def test_identifies_solid_cells_touching_fluid(self):
    fluid = np.ones((5, 5, 5), bool)
    fluid[2, 2, 2] = False          # a single solid cell
    mask = gp.ghost_cell_mask(fluid, halo_width=0)
    self.assertEqual(mask.sum(), 1)
    self.assertTrue(mask[2, 2, 2])

  def test_includes_diagonal_neighbours(self):
    # Ghias section 2.5 uses the 27-point neighbourhood, so a solid cell whose
    # only fluid neighbour is diagonal still counts.
    fluid = np.zeros((3, 3, 3), bool)
    fluid[0, 0, 0] = True
    mask = gp.ghost_cell_mask(fluid, halo_width=0)
    self.assertTrue(mask[1, 1, 1], 'diagonal-only neighbour was missed')

  def test_deep_interior_is_not_a_ghost_cell(self):
    fluid = np.zeros((7, 7, 7), bool)
    fluid[0, :, :] = True
    mask = gp.ghost_cell_mask(fluid, halo_width=0)
    self.assertTrue(mask[1, 3, 3])
    self.assertFalse(mask[3, 3, 3], 'a cell two layers deep was marked')

  def test_dilation_does_not_wrap(self):
    # A solid slab at the low end and fluid at the high end. With a wrapping
    # neighbourhood the cell at index 0 would see the fluid at index -1 and be
    # marked, and its image point would then project right across the domain.
    fluid = np.zeros((6, 3, 3), bool)
    fluid[4:, :, :] = True
    mask = gp.ghost_cell_mask(fluid, halo_width=0)
    self.assertTrue(mask[3, 1, 1], 'the cell adjacent to fluid was missed')
    self.assertFalse(mask[0, 1, 1], 'the dilation wrapped around the axis')

  def test_halo_layers_are_excluded(self):
    fluid = np.ones((9, 9, 9), bool)
    fluid[1, 1, 1] = False          # inside the halo
    fluid[4, 4, 4] = False          # interior
    mask = gp.ghost_cell_mask(fluid, halo_width=2)
    self.assertFalse(mask[1, 1, 1], 'a halo ghost cell was kept')
    self.assertTrue(mask[4, 4, 4])


class BuildTest(absltest.TestCase):

  def test_empty_when_the_body_misses_the_subdomain(self):
    ax = _uniform(8)
    far = sdf.sphere((10.0, 10.0, 10.0), 1.0)
    geom = gp.build(far, (ax, ax, ax), halo_width=2)
    self.assertEqual(geom.num_ghost_points, 0)
    # And applying it is a no-op rather than an error.
    u = jnp.ones((8, 8, 8))
    zeta, gamma = gp.dirichlet_coefficients(geom, 0.0)
    np.testing.assert_array_equal(
        np.asarray(gp.apply(u, geom, zeta, gamma)), np.asarray(u)
    )

  def test_weights_sum_to_one(self):
    ax = _uniform(24)
    geom = gp.build(
        sdf.sphere((0.5, 0.5, 0.5), 0.25), (ax, ax, ax), halo_width=2
    )
    self.assertGreater(geom.num_ghost_points, 0)
    total = geom.alpha.sum(axis=1) + geom.self_weight
    np.testing.assert_allclose(total, 1.0, atol=1e-5)

  def test_corner_indices_are_in_range(self):
    ax = _uniform(20)
    geom = gp.build(
        sdf.cylinder((0.5, 0.5, 0.0), (0.0, 0.0, 1.0), 0.2),
        (ax, ax, ax), halo_width=2,
    )
    n = 20 * 20 * 20
    self.assertTrue((geom.corner_index >= 0).all())
    self.assertTrue((geom.corner_index < n).all())
    self.assertTrue((geom.gc_index >= 0).all())
    self.assertTrue((geom.gc_index < n).all())

  def test_image_points_land_in_the_fluid(self):
    # The construction is only meaningful if the image point is on the fluid
    # side of the surface.
    ax = _uniform(24)
    phi = sdf.sphere((0.5, 0.5, 0.5), 0.25)
    geom = gp.build(phi, (ax, ax, ax), halo_width=2)
    ijk = np.array(np.unravel_index(geom.gc_index, geom.shape)).T
    gc = np.stack([ax[ijk[:, a]] for a in range(3)], axis=-1)
    ip = gc + geom.probe[:, None] * geom.normal * geom.distance[:, None] * -1.0
    # `normal` points into the fluid and the body intercept lies along it, so
    # stepping from the ghost cell toward the surface means moving along +n.
    ip = gc + (geom.probe * geom.distance)[:, None] * geom.normal
    val = np.asarray(sdf.evaluate(phi, jnp.asarray(ip, jnp.float32)))
    self.assertGreater((val > 0).mean(), 0.99)

  def test_probe_is_capped_near_the_array_edge_not_clamped(self):
    # A body close to the edge forces some image points to want to leave the
    # local array. Those cells must have m < 2, and every stencil must still
    # be inside the array -- the TF implementation clamped the index instead,
    # which silently interpolates from the wrong cells.
    ax = _uniform(16)
    geom = gp.build(
        sdf.sphere((0.08, 0.5, 0.5), 0.06), (ax, ax, ax), halo_width=1
    )
    self.assertGreater(geom.num_ghost_points, 0)
    self.assertTrue((geom.probe <= 2.0 + 1e-6).all())
    self.assertTrue((geom.probe >= 1.0).all())
    self.assertTrue((geom.corner_index < 16**3).all())

  def test_summary_reports_capping(self):
    ax = _uniform(16)
    geom = gp.build(
        sdf.sphere((0.08, 0.5, 0.5), 0.06), (ax, ax, ax), halo_width=1
    )
    text = geom.summary()
    self.assertIn('ghost points:', text)
    self.assertIn('reach', text)


class ExactnessTest(parameterized.TestCase):
  """A linear field over a planar boundary must be reproduced exactly."""

  def _run(self, axes, dims, slope, offset, normal_vec, through, bc):
    """Builds a linear field satisfying `bc` and returns (got, want) at GCs.

    The field is `u(x) = offset + slope * n.(x - through)`, which is constant
    on the plane and therefore satisfies a Dirichlet condition with value
    `offset`, and has normal derivative `slope` everywhere.
    """
    phi = sdf.plane(through, normal_vec)
    geom = gp.build(phi, axes, halo_width=1, dims=dims)
    self.assertGreater(geom.num_ghost_points, 0)

    n = np.asarray(normal_vec, np.float64)
    n = n / np.linalg.norm(n)
    mesh = np.meshgrid(*axes, indexing='ij')
    physical = [None, None, None]
    for data_axis, dim in enumerate(dims):
      physical[dim] = mesh[data_axis]
    signed = sum(
        n[d] * (physical[d] - through[d]) for d in range(3)
    )
    u = offset + slope * signed

    if bc == 'dirichlet':
      zeta, gamma = gp.dirichlet_coefficients(geom, offset)
    else:
      zeta, gamma = gp.neumann_coefficients(geom, slope)

    # Wipe exactly the cells the method is responsible for, so it cannot
    # pass by leaving values alone. Ghost cells in the halo keep their value
    # because a real run receives them from the halo exchange, and deep solid
    # cells are never read by any stencil.
    scrambled = u.copy()
    scrambled.reshape(-1)[geom.gc_index] = -999.0
    got = np.asarray(
        gp.apply(jnp.asarray(scrambled, jnp.float64), geom, zeta, gamma,
                 num_sweeps=8)
    )
    idx = np.unravel_index(geom.gc_index, geom.shape)
    # Fallback cells use the nearest-fluid approximation and are documented
    # as first-order, so they are excluded from the exactness claim.
    keep = _reliable(geom)
    return got[idx][keep], u[idx][keep]

  @parameterized.named_parameters(
      ('dirichlet_axis_aligned', 'dirichlet', (0.0, 0.0, 1.0)),
      ('dirichlet_diagonal', 'dirichlet', (1.0, 1.0, 1.0)),
      ('dirichlet_oblique', 'dirichlet', (0.3, -0.7, 1.0)),
      ('neumann_axis_aligned', 'neumann', (0.0, 0.0, 1.0)),
      ('neumann_diagonal', 'neumann', (1.0, 1.0, 1.0)),
      ('neumann_oblique', 'neumann', (0.3, -0.7, 1.0)),
  )
  def test_linear_field_is_exact(self, bc, normal_vec):
    ax = _uniform(20)
    got, want = self._run(
        (ax, ax, ax), (0, 1, 2), slope=2.5, offset=7.0,
        normal_vec=normal_vec, through=(0.5, 0.5, 0.5), bc=bc,
    )
    np.testing.assert_allclose(got, want, atol=_EXACT)

  def test_the_weights_still_partition_unity(self):
    """Moving weight to the intercept must not create or destroy any.

    The trilinear weights sum to one, so a constant field is reproduced
    exactly. The transfer keeps that only if nothing leaks:
    `sum(alpha) + w_self + w_bi == 1` for every cell.
    """
    ax = _uniform(20)
    phi = sdf.plane((0.5, 0.5, 0.5), (0.3, -0.7, 1.0))
    geom = gp.build(phi, (ax, ax, ax), halo_width=1, dims=(0, 1, 2))
    total = (geom.alpha.sum(axis=1) + geom.self_weight + geom.bi_weight)
    np.testing.assert_allclose(total, 1.0, atol=1e-6)

  def test_a_constant_field_is_exact_even_where_anchored(self):
    """Anchoring costs linear accuracy, not constant-field accuracy.

    For a constant field the intercept carries the same value as every other
    node, so the convex combination reproduces it exactly. This is what makes
    the transfer safe to apply unconditionally.
    """
    ax = _uniform(20)
    phi = sdf.plane((0.5, 0.5, 0.5), (0.3, -0.7, 1.0))
    geom = gp.build(phi, (ax, ax, ax), halo_width=1, dims=(0, 1, 2))
    self.assertGreater(int((geom.bi_weight > 1e-6).sum()), 0,
                       'this geometry has no degenerate stencils to test')
    zeta, gamma = gp.dirichlet_coefficients(geom, 5.0)
    u = jnp.full(geom.shape, 5.0, jnp.float64)
    got = np.asarray(gp.apply(u, geom, zeta, gamma, num_sweeps=4))
    idx = np.unravel_index(geom.gc_index, geom.shape)
    np.testing.assert_allclose(got[idx], 5.0, atol=_EXACT)

  def test_a_fully_starved_stencil_takes_the_boundary_value(self):
    """With no usable node at all the ghost cell becomes the surface value.

    That is the right answer rather than a lucky one: with `w_bi == 1` the
    interpolated image-point value *is* the intercept value, and the Dirichlet
    relation then collapses to `phi_GC = Phi`. Before the transfer such a cell
    was determined entirely by other solid cells, with nothing anchoring it.
    """
    geom = gp.GhostPoints(
        gc_index=np.zeros(1, np.int32),
        corner_index=np.zeros((1, 8), np.int32),
        alpha=np.zeros((1, 8), np.float32),
        self_weight=np.zeros(1, np.float32),
        bi_weight=np.ones(1, np.float32),
        zeta_dirichlet=np.array([-1.0], np.float32),
        separation=np.array([2.0], np.float32),
        normal=np.zeros((1, 3), np.float32),
        distance=np.array([1.0], np.float32),
        probe=np.array([2.0], np.float32),
        signed_distance=np.zeros((1, 1, 1), np.float32),
        fallback=np.zeros(1, bool),
        shape=(1, 1, 1),
        num_capped=0,
        num_fallback=0,
        min_probe=2.0,
        max_reach_cells=(0.0, 0.0, 0.0),
    )
    zeta, gamma = gp.dirichlet_coefficients(geom, 3.0)
    got = np.asarray(gp.apply(jnp.zeros(geom.shape, jnp.float64), geom,
                              zeta, gamma))
    np.testing.assert_allclose(got.reshape(-1)[0], 3.0, atol=_EXACT)

  def test_exact_on_a_stretched_grid(self):
    # The enclosing cell and the weights come from a coordinate search, so a
    # graded axis must be just as exact as a uniform one.
    axes = (_stretched(20), _uniform(20), _stretched(18))
    got, want = self._run(
        axes, (0, 1, 2), slope=-1.3, offset=2.0,
        normal_vec=(0.4, 0.2, 1.0), through=(0.5, 0.5, 0.5), bc='dirichlet',
    )
    np.testing.assert_allclose(got, want, atol=_EXACT)

  def test_exact_under_a_permuted_axis_order(self):
    # data_axis_order ('z', 'x', 'y') as the TensorFlow solver used.
    axes = (_uniform(18), _uniform(20), _uniform(22))
    got, want = self._run(
        axes, (2, 0, 1), slope=1.7, offset=-3.0,
        normal_vec=(1.0, 0.5, -0.2), through=(0.5, 0.5, 0.5), bc='dirichlet',
    )
    np.testing.assert_allclose(got, want, atol=_EXACT)

  @parameterized.named_parameters(
      ('m_2_0', 2.0), ('m_1_7', 1.7), ('m_1_5', 1.5), ('m_1_2', 1.2),
  )
  def test_extrapolation_algebra_holds_at_any_probe(self, m):
    """The generalized coefficients, checked against a 1D linear profile.

    An end-to-end capped case is hard to construct reliably -- with
    `halo_width >= 2` the stencil almost always fits and the probe is never
    shortened -- so the algebra that capping relies on is pinned directly.
    `m = 2` is the paper's case and must reduce to `zeta = -1, Gamma = 2*Phi`.
    """
    d, surface, slope = 0.37, 5.0, -2.25
    # u(s) = surface + slope*s, s measured from the intercept along the
    # normal; the ghost cell is at -d and the image point at (m-1)*d.
    u_gc = surface + slope * (-d)
    u_ip = surface + slope * (m - 1.0) * d

    geom = gp.GhostPoints(
        gc_index=np.zeros(1, np.int32),
        corner_index=np.zeros((1, 8), np.int32),
        alpha=np.zeros((1, 8), np.float32),
        self_weight=np.zeros(1, np.float32),
        bi_weight=np.zeros(1, np.float32),
        zeta_dirichlet=np.array([-1.0 / (m - 1.0)], np.float32),
        separation=np.array([m * d], np.float32),
        normal=np.zeros((1, 3), np.float32),
        distance=np.array([d], np.float32),
        probe=np.array([m], np.float32),
        signed_distance=np.zeros((1, 1, 1), np.float32),
        fallback=np.zeros(1, bool),
        shape=(1, 1, 1),
        num_capped=0,
        num_fallback=0,
        min_probe=m,
        max_reach_cells=(0.0, 0.0, 0.0),
    )

    # The scheme amplifies round-off by |zeta| = 1/(m-1), so the tolerance
    # has to scale with it; a fixed bound would simply be a test of m.
    tol = 1e-6 / (m - 1.0)
    zeta, gamma = gp.dirichlet_coefficients(geom, surface)
    np.testing.assert_allclose(
        float(zeta[0] * u_ip + gamma[0]), u_gc, atol=tol
    )
    zeta, gamma = gp.neumann_coefficients(geom, slope)
    np.testing.assert_allclose(
        float(zeta[0] * u_ip + gamma[0]), u_gc, atol=tol
    )

  def test_paper_coefficients_recovered_at_m_two(self):
    # Ghias Eq. 25: zeta = -1 and Gamma = 2*Phi for Dirichlet.
    ax = _uniform(16)
    geom = gp.build(
        sdf.plane((0.5, 0.5, 0.5), (0.0, 0.0, 1.0)), (ax, ax, ax),
        halo_width=2,
    )
    self.assertEqual(geom.num_capped, 0)
    zeta, gamma = gp.dirichlet_coefficients(geom, 3.0)
    np.testing.assert_allclose(np.asarray(zeta), -1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gamma), 6.0, atol=1e-5)

  def test_sweeps_do_not_disturb_an_exact_solution(self):
    # Repeating the update is the explicit stand-in for the paper's coupled
    # solve; it must be idempotent once the answer is right.
    ax = _uniform(20)
    phi = sdf.plane((0.5, 0.5, 0.5), (0.3, -0.7, 1.0))
    geom = gp.build(phi, (ax, ax, ax), halo_width=1)
    n = np.array([0.3, -0.7, 1.0]); n /= np.linalg.norm(n)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    signed = n[0]*(xx-0.5) + n[1]*(yy-0.5) + n[2]*(zz-0.5)
    u = 7.0 + 2.5 * signed
    zeta, gamma = gp.dirichlet_coefficients(geom, 7.0)
    scrambled = u.copy()
    scrambled.reshape(-1)[geom.gc_index] = -999.0
    idx = np.unravel_index(geom.gc_index, geom.shape)
    keep = _reliable(geom)
    prev = None
    for sweeps in (4, 8, 16):
      got = np.asarray(
          gp.apply(jnp.asarray(scrambled), geom, zeta, gamma,
                   num_sweeps=sweeps)
      )[idx][keep]
      if prev is not None:
        np.testing.assert_allclose(
            got, prev, atol=1e-5,
            err_msg='the sweep iteration has not reached a fixed point',
        )
      prev = got


class FallbackTest(absltest.TestCase):
  """The nearest-fluid path, which the exactness tests deliberately exclude.

  Those tests drop fallback cells and everything downstream of them, so
  without this the fallback branch would be unverified.
  """

  def _geometry_with_fallbacks(self):
    # A single halo layer leaves too little room for the full probe on the
    # cells nearest the array edge, which is what triggers the fallback.
    ax = _uniform(20)
    phi = sdf.plane((0.5, 0.5, 0.5), (0.3, -0.7, 1.0))
    geom = gp.build(phi, (ax, ax, ax), halo_width=1)
    self.assertGreater(
        int(geom.fallback.sum()), 0, 'no fallback cells; the test is void'
    )
    return ax, phi, geom

  def test_dirichlet_fallback_takes_the_surface_value(self):
    ax, phi, geom = self._geometry_with_fallbacks()
    surface_value = 7.0
    zeta, gamma = gp.dirichlet_coefficients(geom, surface_value)
    u = jnp.zeros(geom.shape)
    got = np.asarray(gp.apply(u, geom, zeta, gamma))
    idx = np.unravel_index(geom.gc_index, geom.shape)
    np.testing.assert_allclose(
        got[idx][geom.fallback], surface_value, atol=_EXACT,
        err_msg='a fallback cell did not take the boundary value',
    )

  def test_fallback_slope_is_zero_for_dirichlet(self):
    # The mechanism behind the test above: zeta is zeroed so the ghost value
    # is Gamma alone and does not depend on the interpolated field.
    _, _, geom = self._geometry_with_fallbacks()
    zeta, _ = gp.dirichlet_coefficients(geom, 1.0)
    np.testing.assert_allclose(
        np.asarray(zeta)[geom.fallback], 0.0, atol=1e-12
    )
    self.assertTrue(
        (np.abs(np.asarray(zeta)[~geom.fallback]) > 0.0).all(),
        'a non-fallback cell was given a zero slope',
    )

  def test_neumann_fallback_uses_the_nearest_fluid_cell(self):
    # phi_GC = u_nearest - flux * |nearest - GC|, which for a uniform field
    # and zero flux is just the neighbour's value.
    _, _, geom = self._geometry_with_fallbacks()
    zeta, gamma = gp.neumann_coefficients(geom, 0.0)
    u = jnp.full(geom.shape, 4.0)
    got = np.asarray(gp.apply(u, geom, zeta, gamma))
    idx = np.unravel_index(geom.gc_index, geom.shape)
    np.testing.assert_allclose(
        got[idx][geom.fallback], 4.0, atol=_EXACT
    )

  def test_separation_is_positive_everywhere(self):
    _, _, geom = self._geometry_with_fallbacks()
    self.assertTrue((geom.separation > 0.0).all())
    self.assertTrue((geom.probe >= gp._MIN_PROBE - 1e-6).all())  # pylint: disable=protected-access


class CurvedBoundaryTest(absltest.TestCase):
  """A curved boundary is not exact, but must converge at second order."""

  def _error(self, n):
    ax = _uniform(n)
    phi = sdf.sphere((0.5, 0.5, 0.5), 0.25)
    geom = gp.build(phi, (ax, ax, ax), halo_width=1)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.sqrt((xx-0.5)**2 + (yy-0.5)**2 + (zz-0.5)**2)
    # Linear in the radial distance, so Dirichlet value 3.0 on the sphere.
    u = 3.0 + 2.0 * (r - 0.25)
    zeta, gamma = gp.dirichlet_coefficients(geom, 3.0)
    scrambled = u.copy()
    scrambled.reshape(-1)[geom.gc_index] = -999.0
    got = np.asarray(
        gp.apply(jnp.asarray(scrambled, jnp.float64), geom, zeta, gamma,
                 num_sweeps=8)
    )
    idx = np.unravel_index(geom.gc_index, geom.shape)
    keep = _reliable(geom)
    return np.abs(got[idx][keep] - u[idx][keep]).max()

  def test_second_order_convergence(self):
    e1, e2 = self._error(24), self._error(48)
    order = np.log2(e1 / e2)
    self.assertGreater(
        order, 1.6, f'observed order {order:.2f} is below second order'
    )


if __name__ == '__main__':
  absltest.main()


class PartitionIndependenceTest(absltest.TestCase):
  """The same geometry split across devices must give the same answer.

  Three separate bugs in this codebase have been per-replica data silently
  becoming replica 0's copy, or a stencil quietly reading the wrong cells near
  a seam. A ghost-point geometry is per-replica data read through an index
  array, so it is squarely in that class and gets a direct test rather than
  being inferred from single-device correctness.
  """

  def _grid_params(self, cz, nz):
    text = f"""
    computation_shape {{ dim_x: 1 dim_y: 1 dim_z: {cz} }}
    grid_size {{ dim_x: 16 dim_y: 16 dim_z: {nz} }}
    length {{ dim_x: 1.0 dim_y: 1.0 dim_z: 2.0 }}
    halo_width: 2
    dt: 0.1
    data_axis_order: 'xyz'
    """
    return gp_lib.GridParametrization(gp_lib.params_from_text_proto(text))

  def _mesh(self, cz):
    devices = jax.devices()[:cz]
    return jax.sharding.Mesh(
        mesh_utils.create_device_mesh((1, 1, cz), devices=devices),
        axis_names=('x', 'y', 'z'),
    )

  def _field(self, grid_params, mesh, fn):
    """An analytic field laid out exactly as the solver shards it."""
    counts = tuple(mesh.shape[n] for n in mesh.axis_names)
    blocks = []
    for i2 in range(counts[2]):
      coords = gp.local_coordinates(grid_params, (0, 0, i2))
      xx, yy, zz = np.meshgrid(*coords, indexing='ij')
      blocks.append(fn(xx, yy, zz))
    return np.concatenate(blocks, axis=2)

  def _run(self, cz, nz, phi, fn, num_sweeps=1):
    grid_params = self._grid_params(cz, nz)
    mesh = self._mesh(cz)
    sharded, per_replica = gp.build_for_mesh(phi, grid_params, mesh)
    field = self._field(grid_params, mesh, fn)

    spec = jax.sharding.PartitionSpec(*mesh.axis_names)
    specs = gp.partition_specs(mesh)

    def body(u, geom):
      local = gp.squeeze_local(geom)
      zeta, gamma = gp.dirichlet_coefficients(local, 3.0)
      return gp.apply_local(u, local, zeta, gamma, num_sweeps=num_sweeps)

    out = jax.jit(
        shard_map.shard_map(
            body, mesh=mesh, in_specs=(spec, specs), out_specs=spec,
            check_rep=False,
        )
    )(jnp.asarray(field, jnp.float32), sharded)
    total = sum(g.num_ghost_points for g in per_replica)
    return np.asarray(out), grid_params, total

  def test_cylinder_spanning_a_seam_is_partition_independent(self):
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    # A sphere at mid-height, so the body straddles the z split.
    phi = sdf.sphere((0.5, 0.5, 1.0), 0.3)
    fn = lambda x, y, z: (1.0 + x + 2.0 * y - 0.5 * z).astype(np.float32)

    one, gp_one, n_one = self._run(1, 36, phi, fn)
    two, gp_two, n_two = self._run(2, 20, phi, fn)
    self.assertGreater(n_one, 0)
    # The two-device run has extra ghost cells only where the duplicated halo
    # slabs add copies, so the counts need not match exactly; both must be
    # non-trivial.
    self.assertGreater(n_two, 0)

    a = global_layout.strip_halos_global(one, gp_one)
    b = global_layout.strip_halos_global(two, gp_two)
    self.assertEqual(a.shape, b.shape)
    scale = max(float(np.abs(a).max()), 1e-30)
    worst = float(np.abs(a - b).max()) / scale
    self.assertLess(
        worst, 1e-6,
        f'the ghost-point update depends on the partitioning: max relative '
        f'difference {worst:.3e} (the seam is at z={gp_two.core_nz - 1}).',
    )

  def test_geometry_differs_between_replicas(self):
    # Guards the premise. If every replica got the same geometry -- the
    # failure mode that replicated inflow planes produced -- the test above
    # could pass for the wrong reason.
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    grid_params = self._grid_params(2, 20)
    mesh = self._mesh(2)
    _, per_replica = gp.build_for_mesh(
        sdf.sphere((0.5, 0.5, 0.6), 0.25), grid_params, mesh
    )
    counts = [g.num_ghost_points for g in per_replica]
    self.assertNotEqual(
        counts[0], counts[1],
        f'both replicas got {counts[0]} ghost points; the sphere sits off '
        'centre so they should differ',
    )

  def test_padding_entries_are_inert(self):
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    # A body entirely inside one replica: the other has zero ghost points and
    # is pure padding, which must leave its field untouched.
    phi = sdf.sphere((0.5, 0.5, 0.35), 0.2)
    fn = lambda x, y, z: (x + y + z).astype(np.float32)
    grid_params = self._grid_params(2, 20)
    mesh = self._mesh(2)
    _, per_replica = gp.build_for_mesh(phi, grid_params, mesh)
    self.assertIn(0, [g.num_ghost_points for g in per_replica],
                  'expected one replica to hold no ghost points')

    out, _, _ = self._run(2, 20, phi, fn)
    field = self._field(grid_params, mesh, fn)
    empty = [g.num_ghost_points for g in per_replica].index(0)
    nz_local = grid_params.core_nz + 2 * grid_params.halo_width
    sl = slice(empty * nz_local, (empty + 1) * nz_local)
    np.testing.assert_allclose(
        out[:, :, sl], field[:, :, sl], atol=1e-6,
        err_msg='padding entries modified a replica with no ghost points',
    )


class ProtoGeometryTest(absltest.TestCase):
  """Building the signed distance function from configuration."""

  def _parse(self, text):
    message = immersed_boundary_method_pb2.GhostPointMethod()
    text_format.Parse(text, message)
    return message

  def test_cylinder_about_a_non_grid_axis(self):
    # The case the height-field description could never express.
    message = self._parse(
        'geometry { cylinder { base { } axis { x: 1.0 y: 1.0 } radius: 1.0 } }'
    )
    phi = gp.geometry_from_proto(message.geometry)
    # (0,0,5) is 5 from the (1,1,0) axis, so the distance is 5 - 1.
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(phi, jnp.array([[0.0, 0.0, 5.0]], jnp.float32))),
        [4.0], atol=1e-5,
    )

  def test_several_bodies_compose_by_union(self):
    message = self._parse("""
      geometry { sphere { center { x: -2.0 } radius: 1.0 } }
      geometry { sphere { center { x:  2.0 } radius: 1.0 } }
    """)
    phi = gp.geometry_from_proto(message.geometry)
    pts = jnp.array([[-2.0, 0, 0], [2.0, 0, 0], [0.0, 0, 0]], jnp.float32)
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(phi, pts)), [-1.0, -1.0, 1.0], atol=1e-5
    )

  def test_box_rotation_is_applied(self):
    plain = self._parse(
        'geometry { box { center { } half_extents { x: 1 y: 2 z: 3 } } }'
    )
    spun = self._parse(
        'geometry { box { center { } half_extents { x: 1 y: 2 z: 3 }'
        ' rotation_deg { z: 90 } } }'
    )
    pt = jnp.array([[2.5, 0.0, 0.0]], jnp.float32)
    a = float(sdf.evaluate(gp.geometry_from_proto(plain.geometry), pt)[0])
    b = float(sdf.evaluate(gp.geometry_from_proto(spun.geometry), pt)[0])
    self.assertAlmostEqual(a, 1.5, places=5)
    self.assertAlmostEqual(b, 0.5, places=5)   # the long side now faces +x

  def test_height_field_profile_must_be_supplied(self):
    message = self._parse(
        'geometry { height_field { profile: "gaussian" } }'
    )
    with self.assertRaisesRegex(ValueError, 'gaussian'):
      gp.geometry_from_proto(message.geometry)
    phi = gp.geometry_from_proto(
        message.geometry, height_fields={'gaussian': lambda x, y: 0.5 + 0 * x}
    )
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(phi, jnp.array([[0.0, 0.0, 1.5]], jnp.float32))),
        [1.0], atol=1e-6,
    )

  def test_empty_and_unset_geometry_raise(self):
    with self.assertRaisesRegex(ValueError, 'at least one'):
      gp.geometry_from_proto([])
    message = self._parse('geometry { }')
    with self.assertRaisesRegex(ValueError, 'must set one of'):
      gp.geometry_from_proto(message.geometry)

  def test_mixed_scene_matches_per_body_closures(self):
    """Grouping by type must not change the answer, only the graph size."""
    message = self._parse("""
      geometry { sphere { center { x: 0.2 y: 0.3 z: 0.4 } radius: 0.15 } }
      geometry { box { center { x: 0.7 y: 0.2 z: 0.5 }
                       half_extents { x: 0.1 y: 0.2 z: 0.15 }
                       rotation_deg { z: 25.0 } } }
      geometry { box { center { x: 0.3 y: 0.8 z: 0.5 }
                       half_extents { x: 0.08 y: 0.08 z: 0.3 } } }
      geometry { cylinder { base { x: 0.9 y: 0.9 } axis { z: 1.0 }
                            radius: 0.07 } }
      geometry { sphere { center { x: 0.5 y: 0.5 z: 0.9 } radius: 0.1 } }
    """)
    grouped = gp.geometry_from_proto(message.geometry)
    reference = sdf.union(
        sdf.sphere((0.2, 0.3, 0.4), 0.15),
        sdf.box((0.7, 0.2, 0.5), (0.1, 0.2, 0.15), rotation_deg=(0, 0, 25.0)),
        sdf.box((0.3, 0.8, 0.5), (0.08, 0.08, 0.3)),
        sdf.cylinder((0.9, 0.9, 0.0), (0.0, 0.0, 1.0), 0.07),
        sdf.sphere((0.5, 0.5, 0.9), 0.1),
    )
    rng = np.random.default_rng(7)
    pts = jnp.asarray(rng.uniform(0.0, 1.0, size=(400, 3)), jnp.float32)
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(grouped, pts)),
        np.asarray(sdf.evaluate(reference, pts)),
        atol=1e-5,
    )

  def test_a_scene_of_many_bodies_builds_and_is_usable(self):
    """An urban-style scene: the case the closure-per-body union could not do.

    At 1000 bodies the closure union took 80 s to compile, because every body
    added operations to the traced graph. Grouped by type the graph size is
    fixed, so this builds and runs in well under a second.
    """
    message = immersed_boundary_method_pb2.GhostPointMethod()
    rng = np.random.default_rng(3)
    for _ in range(300):
      shape = message.geometry.add().box
      centre = rng.uniform(0.1, 0.9, 3)
      shape.center.x, shape.center.y, shape.center.z = map(float, centre)
      shape.half_extents.x = shape.half_extents.y = 0.02
      shape.half_extents.z = 0.05
      shape.rotation_deg.z = float(rng.uniform(0.0, 90.0))
    for _ in range(150):
      shape = message.geometry.add().cylinder
      base = rng.uniform(0.1, 0.9, 2)
      shape.base.x, shape.base.y = float(base[0]), float(base[1])
      shape.axis.z = 1.0
      shape.radius = 0.015

    phi = gp.geometry_from_proto(message.geometry)
    ax = np.linspace(0.025, 0.975, 20, dtype=np.float32)
    values = np.asarray(sdf.evaluate(phi, sdf.grid_points(ax, ax, ax)))
    self.assertTrue(np.all(np.isfinite(values)))
    # Some of the domain is solid and most of it is not.
    solid = (values < 0.0).mean()
    self.assertGreater(solid, 0.0)
    self.assertLess(solid, 0.5)

    # And the geometry is usable by the precompute, which is the point.
    geom = gp.build(phi, (ax, ax, ax), halo_width=2)
    self.assertGreater(geom.num_ghost_points, 0)

  def test_boxes_without_rotation_match_boxes_rotated_by_zero(self):
    # The builder skips the rotation entirely when no box asks for one;
    # that shortcut must not change the result.
    plain = self._parse(
        'geometry { box { center { x: 0.5 } half_extents'
        ' { x: 0.2 y: 0.3 z: 0.4 } } }'
    )
    explicit = self._parse(
        'geometry { box { center { x: 0.5 } half_extents'
        ' { x: 0.2 y: 0.3 z: 0.4 } rotation_deg { } } }'
    )
    rng = np.random.default_rng(9)
    pts = jnp.asarray(rng.uniform(-1.0, 2.0, size=(200, 3)), jnp.float32)
    np.testing.assert_allclose(
        np.asarray(sdf.evaluate(gp.geometry_from_proto(plain.geometry), pts)),
        np.asarray(
            sdf.evaluate(gp.geometry_from_proto(explicit.geometry), pts)
        ),
        atol=1e-6,
    )

  def test_defaults_match_the_documented_behaviour(self):
    message = immersed_boundary_method_pb2.GhostPointMethod()
    self.assertEqual(
        message.enforcement,
        immersed_boundary_method_pb2.GhostPointMethod.DIRECT,
    )
    self.assertEqual(message.num_sweeps, 2)
    self.assertTrue(message.blend_convection)
    self.assertFalse(message.write_signed_distance)


class EnforcementTest(absltest.TestCase):
  """The two ways the ghost value can reach the solver."""

  def _setup(self):
    ax = _uniform(20)
    phi = sdf.sphere((0.5, 0.5, 0.5), 0.2)
    geom = gp.build(phi, (ax, ax, ax), halo_width=2)
    self.assertGreater(geom.num_ghost_points, 0)
    return geom

  def test_forcing_vanishes_where_direct_has_converged(self):
    # The force is -beta*(u - u_ghost), so once the field already holds the
    # ghost values there is nothing left to drive.
    geom = self._setup()
    zeta, gamma = gp.dirichlet_coefficients(geom, 2.0)
    u = jnp.zeros(geom.shape)
    settled = gp.apply(u, geom, zeta, gamma, num_sweeps=4)
    force = -3.0 * (settled - gp.apply(settled, geom, zeta, gamma,
                                       num_sweeps=4))
    np.testing.assert_allclose(np.asarray(force), 0.0, atol=_EXACT)

  def test_forcing_is_zero_away_from_the_ghost_cells(self):
    geom = self._setup()
    zeta, gamma = gp.dirichlet_coefficients(geom, 2.0)
    u = jnp.full(geom.shape, 0.5)
    target = gp.apply(u, geom, zeta, gamma, num_sweeps=2)
    force = np.asarray(-3.0 * (u - target))
    touched = np.zeros(geom.shape, bool)
    touched.reshape(-1)[geom.gc_index] = True
    np.testing.assert_array_equal(force[~touched], 0.0)
    self.assertGreater(np.abs(force[touched]).max(), 0.0)

  def test_coefficients_for_dispatches_and_rejects_unknown(self):
    geom = self._setup()
    d1 = gp.coefficients_for(geom, 'dirichlet', 1.0)
    d2 = gp.dirichlet_coefficients(geom, 1.0)
    np.testing.assert_allclose(np.asarray(d1[0]), np.asarray(d2[0]))
    with self.assertRaisesRegex(ValueError, 'dirichlet'):
      gp.coefficients_for(geom, 'robin', 1.0)


class _CompileCounter(pylog.Handler):
  """Counts XLA compilations of the jitted cycle.

  Scrapes the same JAX-internal log message as `CycleCompilationTest` in
  `tgv_simulation_test`; if JAX changes it the count falls to zero and the
  assertion fails loudly rather than passing vacuously.
  """

  def __init__(self):
    super().__init__()
    self.count = 0

  def emit(self, record):
    if record.getMessage().startswith('Compiling _shard_map_cycle_body'):
      self.count += 1


_IB_CONFIG = """
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {
  computation_shape { dim_0: 1 dim_1: 1 dim_2: %(cz)d }
  grid_size { dim_0: 16 dim_1: 16 dim_2: %(nz)d }
  length { dim_0: 1.0 dim_1: 1.0 dim_2: 2.0 }
  halo_width: 2 dt: 1e-3 kernel_size: 4
  periodic { dim_0: true dim_1: true dim_2: true }
}
pressure { solver { jacobi { max_iterations: 5 halo_width: 2 omega: 0.67 } } }
thermodynamics { constant_density {} }
density: 1.0
kinematic_viscosity: 1e-3
num_sub_iterations: 3
use_sgs: false
enable_rhie_chow_correction: false
boundary_models {
  ib {
    ghost_point {
      variables { name: "u" value: 0.0 bc: DIRICHLET }
      variables { name: "v" value: 0.0 bc: DIRICHLET }
      variables { name: "w" value: 0.0 bc: DIRICHLET }
      geometry {
        cylinder { base { x: 0.5 y: 0.5 } axis { z: 1.0 } radius: 0.2 }
      }
      write_signed_distance: true
    }
  }
}
"""


class SolverIntegrationTest(absltest.TestCase):
  """The method driving a real simulation, not just the apply in isolation."""

  def _run(self, cz, nz, num_steps=4, checkpoint_dir=None, handler=None):
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(_IB_CONFIG % {'cz': cz, 'nz': nz}, config)
    params = parameters_lib.SwirlLMParameters(config)
    mesh = driver_lib.create_mesh(params)

    phi = gp.geometry_from_proto(
        params.boundary_models.ib.ghost_point.geometry
    )
    geometry = gp.GhostPointGeometry(
        phi, params.grid_params, mesh, write_signed_distance=True
    )
    params.additional_state_partition_specs.update(geometry.partition_specs())

    def init_fn(params, logical_coordinates):
      grid = params.grid_params
      shape = (grid.nx, grid.ny, grid.nz)
      states = {
          'rho': jnp.ones(shape, jnp.float32),
          'u': jnp.ones(shape, jnp.float32),      # uniform flow into the body
          'v': jnp.zeros(shape, jnp.float32),
          'w': jnp.zeros(shape, jnp.float32),
          'p': jnp.zeros(shape, jnp.float32),
      }
      states.update(geometry.initial_states(logical_coordinates))
      return states

    logger = pylog.getLogger('jax._src.interpreters.pxla')
    if handler is not None:
      logger.addHandler(handler)
      logger.setLevel(pylog.DEBUG)
    try:
      state = driver_lib.run_simulation(
          params=params,
          init_fn=init_fn,
          num_steps=num_steps,
          num_steps_per_cycle=2,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=2 if checkpoint_dir else 0,
      )
    finally:
      if handler is not None:
        logger.removeHandler(handler)
    return state, params

  def test_the_boundary_condition_is_actually_imposed(self):
    state, _ = self._run(1, 36)
    u = np.asarray(state['u'])
    ghost = np.asarray(state['ib_boundary']) > 0.5
    self.assertTrue(np.all(np.isfinite(u)))
    self.assertGreater(ghost.sum(), 0)
    # With a Dirichlet value of zero the ghost value is the mirror of the
    # image point, so a positive free stream must leave every ghost cell
    # negative. That is a sharper check than any norm: a sign error, a
    # dropped update or a geometry that missed the body all break it.
    self.assertTrue(
        np.all(u[ghost] < 0.0),
        f'{(u[ghost] >= 0).sum()} of {ghost.sum()} ghost cells are not the '
        'mirror of the flow outside',
    )

  def test_helper_fields_are_present_and_consistent(self):
    state, _ = self._run(1, 36)
    for key in ('ib_interior_mask', 'ib_boundary', 'ib_phi'):
      self.assertIn(key, state)
    mask = np.asarray(state['ib_interior_mask'])
    phi = np.asarray(state['ib_phi'])
    # The mask is the sign of the signed distance, by construction.
    np.testing.assert_array_equal(mask > 0.5, phi > 0.0)
    # A cylinder of radius 0.2 in a 1x1 cross-section: pi*r^2 ~ 12.6%.
    solid = 1.0 - mask.mean()
    self.assertGreater(solid, 0.05)
    self.assertLess(solid, 0.20)

  def test_checkpoint_keeps_the_fields_and_drops_the_internals(self):
    out = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', '/tmp'), 'ib_ckpt'
    )
    shutil.rmtree(out, ignore_errors=True)
    self._run(1, 36, checkpoint_dir=out)
    store = sorted(glob.glob(os.path.join(out, '*.zarr')))[-1]
    keys = set(zarr.open(store, mode='r').array_keys())
    for key in ('ib_interior_mask', 'ib_boundary', 'ib_phi'):
      self.assertIn(key, keys, f'{key} should be available for plotting')
    leaked = [k for k in keys if k.startswith(gp.GEOMETRY_PREFIX)]
    self.assertEmpty(
        leaked, f'solver internals reached the checkpoint: {leaked}'
    )

  def test_the_cycle_still_compiles_once(self):
    # The geometry arrays must not perturb the jit cache key; a static body
    # means one compilation however many cycles run.
    handler = _CompileCounter()
    self._run(1, 36, num_steps=8, handler=handler)
    self.assertEqual(
        handler.count, 1,
        f'the cycle compiled {handler.count} times; the geometry should be '
        'invariant across cycles',
    )

  def test_simulation_is_partition_independent(self):
    if jax.device_count() < 2:
      raise absltest.SkipTest('needs 2 devices')
    one, params_one = self._run(1, 36)
    two, params_two = self._run(2, 20)
    a = global_layout.strip_halos_global(
        np.asarray(one['u']), params_one.grid_params
    )
    b = global_layout.strip_halos_global(
        np.asarray(two['u']), params_two.grid_params
    )
    self.assertEqual(a.shape, b.shape)
    worst = float(np.abs(a - b).max()) / max(float(np.abs(a).max()), 1e-30)
    self.assertLess(
        worst, 1e-6,
        f'the simulation depends on the partitioning: max relative '
        f'difference {worst:.3e}',
    )
