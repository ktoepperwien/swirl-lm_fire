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
"""Tests for the zarr checkpoint layout.

These tests need no devices: `checkpoint.save` takes plain arrays plus a
`GridParametrization` and never touches a mesh, so a multi-device layout is
reproduced by handing it arrays of the multi-device global shape.
"""

import itertools
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from swirl_lm.jax.io import checkpoint
from swirl_lm.jax.utility import grid_parametrization as gp_lib
import xarray as xr


_CONFIG = """
computation_shape {{ dim_x: {cx} dim_y: {cy} dim_z: {cz} }}
grid_size {{ dim_x: {nx} dim_y: {ny} dim_z: {nz} }}
length {{ dim_x: 2e4 dim_y: 1e4 dim_z: 2.4e2 }}
periodic {{ dim_x: false dim_y: false dim_z: true }}
halo_width: {hw}
data_axis_order: '{order}'
dt: 0.5
"""


def _grid_params(
    cx=2, cy=2, cz=1, nx=128, ny=64, nz=6, hw=2, order='xyz'
):
  text = _CONFIG.format(
      cx=cx, cy=cy, cz=cz, nx=nx, ny=ny, nz=nz, hw=hw, order=order
  )
  return gp_lib.GridParametrization(gp_lib.params_from_text_proto(text))


def _sizes(grid_params):
  """Returns (num_cores, local_n, core_n) per array axis."""
  gp = grid_params
  out = []
  for axis in gp.data_axis_order:
    out.append((
        gp.get_axis_entry(gp.cx, gp.cy, gp.cz, axis),
        gp.get_axis_entry(gp.nx, gp.ny, gp.nz, axis),
        gp.get_axis_entry(gp.core_nx, gp.core_ny, gp.core_nz, axis),
    ))
  return tuple(zip(*out))


def _reference(grid_params):
  """Builds a sharded field and its expected interior, independently.

  Each replica evaluates a function of the global index on its own window,
  which is how the solver builds these arrays. Nothing here calls the code
  under test.

  Args:
    grid_params: The grid parametrization.

  Returns:
    A tuple of (sharded, interior, interior_with_halos) float32 arrays.
  """
  hw = grid_params.halo_width
  num_cores, local_n, core_n = _sizes(grid_params)
  f = [num_cores[d] * core_n[d] for d in range(3)]
  strides = (1, 512, 512 * 256)

  def value(index):
    return np.float32(
        1.0 + sum(strides[d] * index[d] for d in range(3))
    )

  sharded = np.zeros(
      [num_cores[d] * local_n[d] for d in range(3)], dtype=np.float32
  )
  for cores in itertools.product(*[range(c) for c in num_cores]):
    for local in itertools.product(*[range(n) for n in local_n]):
      out = tuple(cores[d] * local_n[d] + local[d] for d in range(3))
      sharded[out] = value(
          tuple(cores[d] * core_n[d] + local[d] - hw for d in range(3))
      )

  interior = np.zeros(f, dtype=np.float32)
  for index in itertools.product(*[range(n) for n in f]):
    interior[index] = value(index)

  with_halos = np.zeros([n + 2 * hw for n in f], dtype=np.float32)
  for index in itertools.product(*[range(n + 2 * hw) for n in f]):
    with_halos[index] = value(tuple(i - hw for i in index))

  return sharded, interior, with_halos


class CheckpointLayoutTest(parameterized.TestCase):

  def _save(self, states, grid_params, **kwargs):
    path = os.path.join(self.create_tempdir().full_path, 'step_000007.zarr')
    checkpoint.save(states, grid_params, step=7, path=path, **kwargs)
    return path

  def test_full_fields_share_grid_dims_with_coordinates(self):
    gp = _grid_params()
    sharded, interior, _ = _reference(gp)
    path = self._save(
        {
            'u': sharded,
            'v': sharded + np.float32(0.5),
            'simulation_time': np.float32(3.5),
        },
        gp,
        sim_time=3.5,
    )
    ds = xr.open_zarr(path)

    # The strongest guard against the old layout assumption: with only full
    # fields saved, no private dimensions may appear at all.
    self.assertEqual(set(ds.sizes), {'x', 'y', 'z'})
    self.assertEqual(ds['u'].dims, ('x', 'y', 'z'))
    self.assertEqual(ds['u'].dims, ds['v'].dims)
    # The duplicated interior seams are gone: 248/120, not 256/128.
    self.assertEqual(dict(ds.sizes), {'x': 248, 'y': 120, 'z': 2})
    np.testing.assert_array_equal(ds['u'].values, interior)
    self.assertEqual(ds['u'].dtype, np.float32)
    for axis, name in enumerate('xyz'):
      np.testing.assert_allclose(
          ds[name].values, np.asarray(gp.global_xyz[axis])
      )

  def test_records_layout_attributes(self):
    gp = _grid_params()
    sharded, _, _ = _reference(gp)
    path = self._save({'u': sharded}, gp)
    ds = xr.open_zarr(path)

    self.assertEqual(
        list(ds['u'].attrs[checkpoint.GLOBAL_SHAPE_ATTR]), [256, 128, 6]
    )
    self.assertEqual(
        list(ds['u'].attrs[checkpoint.STRIPPED_AXES_ATTR]), [0, 1, 2]
    )
    self.assertEqual(ds.attrs['format_version'], checkpoint.FORMAT_VERSION)
    self.assertEqual(ds.attrs['step'], 7)

  def test_scalar_round_trips(self):
    gp = _grid_params()
    sharded, _, _ = _reference(gp)
    path = self._save(
        {'u': sharded, 'simulation_time': np.float32(2.5)}, gp
    )
    ds = xr.open_zarr(path)

    self.assertEqual(ds['simulation_time'].shape, ())
    self.assertEqual(float(ds['simulation_time']), 2.5)

  def test_broadcastable_scale_factor(self):
    """A stretched-grid scale factor is (cx * nx, cy, cz) globally."""
    gp = _grid_params()
    h0 = np.arange(256 * 2 * 1, dtype=np.float32).reshape(256, 2, 1)
    path = self._save({'h0': h0}, gp)
    ds = xr.open_zarr(path)

    # Only the axis it varies along carries the sharded layout.
    self.assertEqual(ds['h0'].shape, (248, 2, 1))
    self.assertEqual(ds['h0'].dims, ('x', 'y_2', 'z_1'))
    self.assertEqual(list(ds['h0'].attrs[checkpoint.STRIPPED_AXES_ATTR]), [0])

  def test_boundary_condition_plane(self):
    """A BC state has a thin normal axis and sharded in-plane axes."""
    gp = _grid_params()
    bc = np.ones((3, 128, 6), dtype=np.float32)
    path = self._save({'bc_u_0_0': bc}, gp)
    ds = xr.open_zarr(path)

    self.assertEqual(ds['bc_u_0_0'].shape, (3, 120, 2))
    self.assertEqual(ds['bc_u_0_0'].dims, ('x_3', 'y', 'z'))
    self.assertEqual(
        list(ds['bc_u_0_0'].attrs[checkpoint.STRIPPED_AXES_ATTR]), [1, 2]
    )

  def test_shared_private_dims_are_not_fragmented(self):
    """Equally shaped variables must share dimensions, so they stay usable."""
    gp = _grid_params()
    bc = np.ones((3, 128, 6), dtype=np.float32)
    path = self._save({'bc_u_0_0': bc, 'bc_v_0_0': bc * 2}, gp)
    ds = xr.open_zarr(path)

    self.assertEqual(ds['bc_u_0_0'].dims, ds['bc_v_0_0'].dims)
    self.assertEqual(
        float((ds['bc_v_0_0'] - ds['bc_u_0_0']).max()), 1.0
    )

  def test_non_xyz_data_axis_order(self):
    gp = _grid_params(order='zxy')
    sharded, interior, _ = _reference(gp)
    self.assertEqual(sharded.shape, (6, 256, 128))

    path = self._save({'u': sharded}, gp)
    ds = xr.open_zarr(path)

    self.assertEqual(ds['u'].dims, ('z', 'x', 'y'))
    self.assertEqual(ds['u'].shape, (2, 248, 120))
    np.testing.assert_array_equal(ds['u'].values, interior)
    # The x coordinate stays the x coordinate, whatever the array order.
    np.testing.assert_allclose(
        ds['x'].values, np.asarray(gp.global_xyz[1])
    )
    self.assertLen(ds['x'].values, 248)

  def test_exclude_halos_false_keeps_outer_halos(self):
    gp = _grid_params()
    sharded, _, with_halos = _reference(gp)
    path = self._save({'u': sharded}, gp, exclude_halos=False)
    ds = xr.open_zarr(path)

    self.assertEqual(dict(ds.sizes), {'x': 252, 'y': 124, 'z': 6})
    np.testing.assert_array_equal(ds['u'].values, with_halos)
    for axis, name in enumerate('xyz'):
      np.testing.assert_allclose(
          ds[name].values, np.asarray(gp.global_xyz_with_halos[axis])
      )

  def test_thin_axis_is_not_confused_with_the_interior(self):
    """`f_x == halo_width + 1` must not make a BC normal axis look stripped."""
    gp = _grid_params(cx=1, nx=7)
    self.assertEqual(gp.core_nx, 3)  # f_x == 3 == halo_width + 1
    bc = np.ones((3, 128, 6), dtype=np.float32)
    path = self._save({'bc_u_0_0': bc}, gp)
    ds = xr.open_zarr(path)

    self.assertEqual(ds['bc_u_0_0'].shape[0], 3)
    self.assertNotIn(
        0, list(ds['bc_u_0_0'].attrs[checkpoint.STRIPPED_AXES_ATTR])
    )

  def test_single_device_is_a_plain_interior_slice(self):
    gp = _grid_params(cx=1, cy=1, cz=1, nx=16, ny=16, nz=16)
    sharded, _, _ = _reference(gp)
    path = self._save({'u': sharded}, gp)
    ds = xr.open_zarr(path)

    self.assertEqual(ds['u'].shape, (12, 12, 12))
    self.assertEqual(ds['u'].dims, ('x', 'y', 'z'))
    np.testing.assert_array_equal(
        ds['u'].values, sharded[2:-2, 2:-2, 2:-2]
    )


class CheckpointLoadTest(absltest.TestCase):

  def test_load_returns_numpy_with_metadata(self):
    gp = _grid_params()
    sharded, interior, _ = _reference(gp)
    path = os.path.join(self.create_tempdir().full_path, 'step_000007.zarr')
    checkpoint.save(
        {'u': sharded, 'simulation_time': np.float32(3.5)},
        gp, step=7, path=path, sim_time=3.5,
    )

    states, metadata = checkpoint.load(path)

    self.assertIsInstance(states['u'], np.ndarray)
    self.assertEqual(states['u'].shape, (248, 120, 2))
    self.assertEqual(states['u'].dtype, np.float32)
    np.testing.assert_array_equal(states['u'], interior)
    self.assertEqual(metadata['step'], 7)
    self.assertEqual(
        list(metadata['variable_attrs']['u'][checkpoint.GLOBAL_SHAPE_ATTR]),
        [256, 128, 6],
    )

  def test_load_selected_fields(self):
    gp = _grid_params()
    sharded, _, _ = _reference(gp)
    path = os.path.join(self.create_tempdir().full_path, 'step_000007.zarr')
    checkpoint.save({'u': sharded, 'v': sharded}, gp, step=7, path=path)

    states, _ = checkpoint.load(path, fields=['v'])
    self.assertEqual(list(states), ['v'])


if __name__ == '__main__':
  absltest.main()
