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
"""Tests for the sharded/contiguous grid layout conversions.

These tests need no devices: the conversions are host-side NumPy and never
touch a mesh.
"""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from swirl_lm.jax.utility import global_layout
from swirl_lm.jax.utility import grid_parametrization as gp_lib


_CONFIG = """
computation_shape {{ dim_x: {cx} dim_y: {cy} dim_z: {cz} }}
grid_size {{ dim_x: {nx} dim_y: {ny} dim_z: {nz} }}
length {{ dim_x: 2e4 dim_y: 1e4 dim_z: 2.4e2 }}
periodic {{ dim_x: {px} dim_y: {py} dim_z: {pz} }}
halo_width: {hw}
data_axis_order: '{order}'
dt: 0.5
"""


def _grid_params(
    cx=2, cy=2, cz=1, nx=128, ny=64, nz=6, hw=2,
    px=False, py=False, pz=True, order='xyz',
):
  """Builds a GridParametrization from the template config."""
  text = _CONFIG.format(
      cx=cx, cy=cy, cz=cz, nx=nx, ny=ny, nz=nz, hw=hw,
      px=str(px).lower(), py=str(py).lower(), pz=str(pz).lower(),
      order=order,
  )
  return gp_lib.GridParametrization(gp_lib.params_from_text_proto(text))


def _reference_arrays(grid_params):
  """Builds a sharded array and its expected conversions, independently.

  The sharded array is built the way the solver builds it: each replica
  evaluates a global function on its own window of the extended index space,
  `[i_core * core_n - hw, i_core * core_n - hw + local_n)`. Periodic axes wrap,
  which is what halo exchange produces. Nothing here uses the functions under
  test, so the expectations are not tautological.

  Args:
    grid_params: The grid parametrization.

  Returns:
    A tuple of (sharded, interior, interior_with_halos) float32 arrays.
  """
  hw = grid_params.halo_width
  sizes = [global_layout._axis_sizes(grid_params, d) for d in range(3)]  # pylint: disable=protected-access
  num_cores = [s[0] for s in sizes]
  local_n = [s[1] for s in sizes]
  core_n = [s[2] for s in sizes]
  periodic = [s[3] for s in sizes]
  f = [num_cores[d] * core_n[d] for d in range(3)]

  # Distinct, exactly representable float32 values keyed on the global index.
  strides = (1, 512, 512 * 256)

  def value(index):
    total = 0.0
    for d in range(3):
      i = index[d] % f[d] if periodic[d] else index[d]
      total += strides[d] * i
    return np.float32(total + 1.0)

  sharded = np.zeros(
      [num_cores[d] * local_n[d] for d in range(3)], dtype=np.float32
  )
  for cores in itertools.product(*[range(c) for c in num_cores]):
    for local in itertools.product(*[range(n) for n in local_n]):
      out = tuple(cores[d] * local_n[d] + local[d] for d in range(3))
      glob = tuple(cores[d] * core_n[d] + local[d] - hw for d in range(3))
      sharded[out] = value(glob)

  interior = np.zeros(f, dtype=np.float32)
  for index in itertools.product(*[range(n) for n in f]):
    interior[index] = value(index)

  with_halos = np.zeros([n + 2 * hw for n in f], dtype=np.float32)
  for index in itertools.product(*[range(n + 2 * hw) for n in f]):
    with_halos[index] = value(tuple(i - hw for i in index))

  return sharded, interior, with_halos


class StrippedAxesTest(absltest.TestCase):

  def test_identifies_full_field_axes(self):
    gp = _grid_params()
    field = np.zeros((256, 128, 6), dtype=np.float32)
    self.assertEqual(global_layout.stripped_axes_for(field, gp), (0, 1, 2))

  def test_passes_through_thin_and_broadcast_axes(self):
    gp = _grid_params()
    # A boundary condition state: thin normal axis, sharded in-plane axes.
    bc = np.zeros((3, 128, 6), dtype=np.float32)
    self.assertEqual(global_layout.stripped_axes_for(bc, gp), (1, 2))
    # A stretched-grid scale factor: (cx * nx, cy, cz).
    h0 = np.zeros((256, 2, 1), dtype=np.float32)
    self.assertEqual(global_layout.stripped_axes_for(h0, gp), (0,))

  def test_already_stripped_array_is_left_alone(self):
    gp = _grid_params()
    interior = np.zeros((248, 120, 2), dtype=np.float32)
    self.assertEqual(global_layout.stripped_axes_for(interior, gp), ())
    np.testing.assert_array_equal(
        global_layout.strip_halos_global(interior, gp), interior
    )

  def test_non_3d_arrays_are_left_alone(self):
    gp = _grid_params()
    scalar = np.float32(2.5)
    self.assertEqual(global_layout.stripped_axes_for(np.asarray(scalar), gp), ())
    np.testing.assert_array_equal(
        global_layout.strip_halos_global(np.asarray(scalar), gp), scalar
    )


class StripHalosTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('multi_device', dict()),
      ('single_device', dict(cx=1, cy=1, cz=1)),
      ('zxy_order', dict(order='zxy')),
      ('all_periodic', dict(px=True, py=True, pz=True)),
  )
  def test_strip_matches_reference(self, overrides):
    gp = _grid_params(**overrides)
    sharded, interior, with_halos = _reference_arrays(gp)

    np.testing.assert_array_equal(
        global_layout.strip_halos_global(sharded, gp), interior
    )
    np.testing.assert_array_equal(
        global_layout.strip_halos_global(sharded, gp, include_halos=True),
        with_halos,
    )

  def test_strip_lengths_match_the_grid_coordinates(self):
    gp = _grid_params()
    sharded, _, _ = _reference_arrays(gp)

    stripped = global_layout.strip_halos_global(sharded, gp)
    extended = global_layout.strip_halos_global(sharded, gp, True)
    for axis in range(3):
      self.assertLen(np.asarray(gp.global_xyz[axis]), stripped.shape[axis])
      self.assertLen(
          np.asarray(gp.global_xyz_with_halos[axis]), extended.shape[axis]
      )

  def test_preserves_dtype(self):
    gp = _grid_params()
    for dtype in (np.float32, np.float64, np.int32):
      sharded = np.ones((256, 128, 6), dtype=dtype)
      self.assertEqual(
          global_layout.strip_halos_global(sharded, gp).dtype, dtype
      )


class InflateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('multi_device', dict()),
      ('single_device', dict(cx=1, cy=1, cz=1)),
      ('zxy_order', dict(order='zxy')),
  )
  def test_round_trip_with_halos_is_exact(self, overrides):
    """A store written with halos restores the sharded array exactly."""
    gp = _grid_params(**overrides)
    sharded, _, _ = _reference_arrays(gp)

    stripped = global_layout.strip_halos_global(sharded, gp, include_halos=True)
    restored = global_layout.inflate_to_global(
        stripped,
        gp,
        target_shape=sharded.shape,
        stripped_axes=global_layout.stripped_axes_for(sharded, gp),
    )
    np.testing.assert_array_equal(restored, sharded)

  def test_round_trip_interior_only_is_exact_when_periodic(self):
    """With every axis periodic, wrapping reproduces all halos."""
    gp = _grid_params(px=True, py=True, pz=True)
    sharded, _, _ = _reference_arrays(gp)

    stripped = global_layout.strip_halos_global(sharded, gp)
    restored = global_layout.inflate_to_global(
        stripped,
        gp,
        target_shape=sharded.shape,
        stripped_axes=global_layout.stripped_axes_for(sharded, gp),
    )
    np.testing.assert_array_equal(restored, sharded)

  def test_round_trip_interior_only_reconstructs_seams(self):
    """Interior seams are exact even when the outer halos cannot be."""
    gp = _grid_params()
    sharded, _, _ = _reference_arrays(gp)
    hw = gp.halo_width

    stripped = global_layout.strip_halos_global(sharded, gp)
    restored = global_layout.inflate_to_global(
        stripped,
        gp,
        target_shape=sharded.shape,
        stripped_axes=global_layout.stripped_axes_for(sharded, gp),
    )
    self.assertEqual(restored.shape, sharded.shape)
    # Everything except the outer halo slabs of the non-periodic axes must
    # match, which includes every duplicated interior seam.
    inner = restored[hw:-hw, hw:-hw, :]
    np.testing.assert_array_equal(inner, sharded[hw:-hw, hw:-hw, :])

  def test_existing_supplies_outer_halos(self):
    """Passing the initialized array back makes the round trip exact."""
    gp = _grid_params()
    sharded, _, _ = _reference_arrays(gp)

    stripped = global_layout.strip_halos_global(sharded, gp)
    restored = global_layout.inflate_to_global(
        stripped,
        gp,
        target_shape=sharded.shape,
        stripped_axes=global_layout.stripped_axes_for(sharded, gp),
        existing=sharded,
    )
    np.testing.assert_array_equal(restored, sharded)

  def test_thin_axis_passes_through(self):
    gp = _grid_params()
    bc = np.arange(3 * 128 * 6, dtype=np.float32).reshape(3, 128, 6)
    axes = global_layout.stripped_axes_for(bc, gp)
    stripped = global_layout.strip_halos_global(bc, gp)
    self.assertEqual(stripped.shape, (3, 120, 2))

    restored = global_layout.inflate_to_global(
        stripped, gp, target_shape=bc.shape, stripped_axes=axes
    )
    self.assertEqual(restored.shape, bc.shape)
    # Inflating is a right inverse of stripping: the unique points survive,
    # and the thin normal axis is carried through untouched. The original's
    # `[hw:-hw]` region cannot be compared directly, since it spans the
    # duplicated seam that stripping discards.
    np.testing.assert_array_equal(
        global_layout.strip_halos_global(restored, gp), stripped
    )

  def test_rejects_unexpected_length(self):
    gp = _grid_params()
    bad = np.zeros((100, 120, 2), dtype=np.float32)
    with self.assertRaisesRegex(ValueError, 'Axis 0 has length 100'):
      global_layout.inflate_to_global(
          bad, gp, target_shape=(256, 128, 6), stripped_axes=(0, 1, 2)
      )

  def test_rejects_unconverted_axis(self):
    gp = _grid_params()
    bad = np.zeros((248, 120, 2), dtype=np.float32)
    with self.assertRaisesRegex(ValueError, 'not recorded as converted'):
      global_layout.inflate_to_global(
          bad, gp, target_shape=(256, 128, 6), stripped_axes=(1, 2)
      )


if __name__ == '__main__':
  absltest.main()
