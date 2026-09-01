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
"""Checkpoint utilities: save/load simulation state as xarray/zarr.

This module provides a clean interface for persisting simulation states.
Each state dict `{name: jax.Array}` is stored as an `xarray.Dataset` in
zarr format, with named dimensions and physical coordinates from the grid.

Key features:
  - Named dimensions (`x`, `y`, `z`) with physical coordinate values.
  - Metadata (step, time, grid config) stored in zarr attrs.
  - Halo-aware: saves only interior points by default.
  - Selective field loading via xarray lazy access.
  - Cloud-compatible (zarr works with local, GCS, etc.).

Usage:
  ```python
  from swirl_lm.jax.io import checkpoint

  # Save.
  checkpoint.save(states, grid_params, step=100, path='/tmp/ckpt_100.zarr')

  # Load.
  states, metadata = checkpoint.load('/tmp/ckpt_100.zarr')
  ```
"""


from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.utility import grid_parametrization as gp_lib
import xarray as xr


def _interior_coords(
    grid_params: gp_lib.GridParametrization,
) -> dict[str, np.ndarray]:
  """Builds physical coordinate arrays for the interior (no halos).

  Note: `global_xyz` already excludes halo points.

  Args:
    grid_params: Grid parametrization with global coordinate info.

  Returns:
    Dict mapping dimension name ('x', 'y', 'z') to 1D coordinate arrays.
  """
  coords = {}
  for i, axis_name in enumerate(grid_params.data_axis_order):
    coords[axis_name] = np.asarray(grid_params.global_xyz[i])
  return coords


def _data_axis_dims(
    grid_params: gp_lib.GridParametrization,
) -> tuple[str, ...]:
  """Returns dimension names in data axis order."""
  return tuple(grid_params.data_axis_order)


def save(
    states: dict[str, jax.Array],
    grid_params: gp_lib.GridParametrization,
    step: int,
    path: str,
    sim_time: float = 0.0,
    exclude_halos: bool = True,
    fields: Sequence[str] | None = None,
) -> None:
  """Saves simulation state as an xarray Dataset in zarr format.

  Args:
    states: Flow field variables, e.g. {'rho': array, 'u': array, ...}.
    grid_params: Grid parametrization for coordinate metadata.
    step: Current simulation step number.
    path: Output path for the zarr store (directory).
    sim_time: Current simulation time.
    exclude_halos: If True, strip halo points from all fields. Default True.
    fields: Optional list of field names to save. If None, saves all.
  """
  hw = grid_params.halo_width
  coords = _interior_coords(grid_params)
  dims = _data_axis_dims(grid_params)

  keys = fields if fields is not None else list(states.keys())
  data_vars = {}
  for name in keys:
    arr = np.asarray(states[name])
    if exclude_halos and hw > 0:
      s = slice(hw, -hw)
      arr = arr[s, s, s]
    data_vars[name] = xr.DataArray(arr, dims=dims, coords=coords)

  ds = xr.Dataset(data_vars)
  ds.attrs['step'] = step
  ds.attrs['sim_time'] = sim_time
  ds.attrs['halo_width'] = hw
  ds.attrs['nx'] = grid_params.nx
  ds.attrs['ny'] = grid_params.ny
  ds.attrs['nz'] = grid_params.nz
  ds.attrs['lx'] = grid_params.lx
  ds.attrs['ly'] = grid_params.ly
  ds.attrs['lz'] = grid_params.lz
  ds.attrs['dt'] = grid_params.dt

  ds.to_zarr(path, mode='w')


def load(
    path: str,
    fields: Sequence[str] | None = None,
) -> tuple[dict[str, jax.Array], dict[str, object]]:
  """Loads simulation state from a zarr checkpoint.

  Args:
    path: Path to the zarr store.
    fields: Optional list of field names to load. If None, loads all.

  Returns:
    A tuple of (states, metadata) where states is a dict of JAX arrays and
    metadata contains step, sim_time, and grid info from attrs.
  """
  ds = xr.open_zarr(path)
  keys = fields if fields is not None else list(ds.data_vars)
  states = {name: jnp.array(ds[name].values) for name in keys}

  metadata = dict(ds.attrs)
  ds.close()
  return states, metadata


def load_as_dataset(path: str) -> xr.Dataset:
  """Loads a checkpoint as an xarray Dataset for interactive analysis.

  This is useful for Colab/notebook workflows where you want to use
  xarray's built-in plotting and analysis tools directly.

  Args:
    path: Path to the zarr store.

  Returns:
    The xarray Dataset with all variables, coordinates, and metadata.
  """
  return xr.open_zarr(path)
