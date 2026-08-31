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

# Copyright 2023 Google LLC
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
"""Data class for loading and accessing longwave optical properties (JAX).

JAX port of `swirl_lm.physics.radiation.optics.lookup_gas_optics_longwave`.
"""


import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4 as nc
from swirl_lm.jax.physics.radiation.optics import data_loader_base as loader
from swirl_lm.jax.physics.radiation.optics import lookup_gas_optics_base as gas_base


@dataclasses.dataclass(frozen=True)
class LookupGasOpticsLongwave(gas_base.AbstractLookupGasOptics):
  """Lookup tables of gases' optical properties in the longwave bands."""

  # Planck fraction `(n_t_ref, n_p_ref, n_eta, n_gpt)`.
  planck_fraction: jax.Array
  # Number of reference temperatures, for Planck source calculations.
  n_t_plnk: int
  # reference temperatures for Planck source calculations `(n_t_plnk)`.
  t_planck: jax.Array
  # total Planck source for each band `(n_bnd, n_t_plnk)`.
  totplnk: jax.Array

  @classmethod
  def _load_data(
      cls,
      ds: nc.Dataset,
      tables: loader.TensorMap,
      dims: loader.DimensionMap,
  ) -> dict[str, Any]:
    """Preprocesses the RRTMGP longwave gas optics data.

    Args:
      ds: The original netCDF Dataset containing the RRTMGP longwave optics
        data.
      tables: The extracted data as a dictionary of JAX arrays.
      dims: A dictionary containing dimension information for the tables.

    Returns:
      A dictionary containing dimension information and the preprocessed RRTMGP
      data as JAX arrays.
    """
    data = super()._load_data(ds, tables, dims)
    data['n_t_plnk'] = dims['temperature_Planck']
    data['planck_fraction'] = tables['plank_fraction']
    # Similarly to the original RRTM Fortran code, here we assume that
    # temperature minimum and maximum are the same for the absorption
    # coefficient grid and the Planck grid and the Planck grid is equally
    # spaced.
    data['t_planck'] = jnp.linspace(
        data['temperature_ref_min'],
        data['temperature_ref_max'],
        data['n_t_plnk'],
    )
    data['totplnk'] = tables['totplnk']
    return data

  @classmethod
  def from_nc_file(cls, path: str) -> 'LookupGasOpticsLongwave':  # pyrefly: ignore[bad-override]
    """Instantiates a `LookupGasOpticsLongwave` object from zipped netCDF file.

    The compressed file should be netCDF parsable and contain the RRTMGP
    absorprtion coefficient lookup table for the longwave bands as well as all
    the auxiliary reference tables required to index into the lookup table.

    Args:
      path: The full path of the zipped netCDF file containing the longwave
        absorption coefficient lookup table.

    Returns:
      A `LookupGasOpticsLongwave` object.
    """
    ds, tables, dims = cls._parse_nc_file(path)
    kwargs = cls._load_data(ds, tables, dims)
    return cls(**kwargs)
