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
"""Data class for loading and accessing shortwave optical properties (JAX).

JAX port of `swirl_lm.physics.radiation.optics.lookup_gas_optics_shortwave`.
"""


import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4 as nc
from swirl_lm.jax.physics.radiation.optics import data_loader_base as loader
from swirl_lm.jax.physics.radiation.optics import lookup_gas_optics_base as gas_base


@dataclasses.dataclass(frozen=True)
class LookupGasOpticsShortwave(gas_base.AbstractLookupGasOptics):
  """Lookup table of gases' optical properties in the shortwave bands."""

  # Total solar irradiation.
  solar_src_tot: float
  # Relative solar source contribution from each `g-point` `(n_gpt)`.
  solar_src_scaled: jax.Array
  # Rayleigh absorption coefficient for lower atmosphere `(n_t_ref, n_eta,
  # n_gpt)`.
  rayl_lower: jax.Array
  # Rayleigh absorption coefficient for upper atmosphere `(n_t_ref, n_eta,
  # n_gpt)`.
  rayl_upper: jax.Array

  @classmethod
  def _load_data(
      cls,
      ds: nc.Dataset,
      tables: loader.TensorMap,
      dims: loader.DimensionMap,
  ) -> dict[str, Any]:
    """Preprocesses the RRTMGP shortwave gas optics data.

    Args:
      ds: The original netCDF Dataset containing the RRTMGP shortwave optics
        data.
      tables: The extracted data as a dictionary of JAX arrays.
      dims: A dictionary containing dimension information for the tables.

    Returns:
      A dictionary containing dimension information and the preprocessed RRTMGP
      data as JAX arrays.
    """
    data = super()._load_data(ds, tables, dims)
    solar_src = tables['solar_source_quiet']
    data['solar_src_tot'] = jnp.sum(solar_src)
    data['solar_src_scaled'] = solar_src / data['solar_src_tot']
    data['rayl_lower'] = tables['rayl_lower']
    data['rayl_upper'] = tables['rayl_upper']
    return data

  @classmethod
  def from_nc_file(  # pyrefly: ignore[bad-override]
      cls, path: str
  ) -> 'LookupGasOpticsShortwave':
    """Instantiates a `LookupGasOpticsShortwave` object from zipped netCDF file.

    The compressed file should be netCDF parsable and contain the RRTMGP
    absorprtion coefficient lookup table for the shortwave bands as well as all
    the auxiliary reference tables required to index into the lookup table.

    Args:
      path: The full path of the zipped netCDF file containing the shortwave
        absorption coefficient lookup table.

    Returns:
      A `LookupGasOpticsShortwave` object.
    """
    ds, tables, dims = cls._parse_nc_file(path)
    kwargs = cls._load_data(ds, tables, dims)
    return cls(**kwargs)
