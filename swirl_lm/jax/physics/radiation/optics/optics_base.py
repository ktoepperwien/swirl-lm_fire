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

"""Abstract base class defining the interface of an optics scheme (JAX).

JAX port of `swirl_lm.physics.radiation.optics.optics_base`.
"""


import abc
from typing import Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.numerics import interpolation
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types
from swirl_lm.physics.radiation.config import radiative_transfer_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


class OpticsScheme(metaclass=abc.ABCMeta):
  """Abstract base class for optics scheme."""

  _EPSILON = 1e-6

  def __init__(
      self,
      params: radiative_transfer_pb2.OpticsParameters,
      g_dim: int,
      halos: int,
  ):
    self._g_dim = g_dim
    self._halos = halos
    self._face_interp_scheme_order = params.face_interp_scheme_order
    self.cloud_optics_lw = None
    self.cloud_optics_sw = None
    self.gas_optics_lw = None
    self.gas_optics_sw = None

  @abc.abstractmethod
  def compute_lw_optical_properties(
      self,
      pressure: ScalarField,
      temperature: ScalarField,
      molecules: ScalarField,
      igpt: jax.Array,
      vmr_fields: Optional[dict[int, ScalarField]] = None,
      cloud_r_eff_liq: Optional[ScalarField] = None,
      cloud_path_liq: Optional[ScalarField] = None,
      cloud_r_eff_ice: Optional[ScalarField] = None,
      cloud_path_ice: Optional[ScalarField] = None,
  ) -> ScalarFieldMap:
    """Computes the monochromatic longwave optical properties.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m**2].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m**2].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m**2].

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The longwave optical depth.
        'ssa': The longwave single-scattering albedo.
        'asymmetry_factor': The longwave asymmetry factor.
    """

  @abc.abstractmethod
  def compute_sw_optical_properties(
      self,
      pressure: ScalarField,
      temperature: ScalarField,
      molecules: ScalarField,
      igpt: jax.Array,
      vmr_fields: Optional[dict[int, ScalarField]] = None,
      cloud_r_eff_liq: Optional[ScalarField] = None,
      cloud_path_liq: Optional[ScalarField] = None,
      cloud_r_eff_ice: Optional[ScalarField] = None,
      cloud_path_ice: Optional[ScalarField] = None,
  ) -> ScalarFieldMap:
    """Computes the monochromatic shortwave optical properties.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m**2].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m**2].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m**2].

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The shortwave optical depth.
        'ssa': The shortwave single-scattering albedo.
        'asymmetry_factor': The shortwave asymmetry factor.
    """

  @abc.abstractmethod
  def compute_planck_sources(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      pressure: ScalarField,
      temperature: ScalarField,
      igpt: jax.Array,
      vmr_fields: Optional[dict[int, ScalarField]] = None,
      sfc_temperature: Optional[ScalarField] = None,
  ) -> ScalarFieldMap:
    """Computes the Planck sources used in the longwave problem.

    Args:
      mesh: The JAX device mesh for multi-device communication.
      grid_params: Grid parametrization.
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      sfc_temperature: An optional 2D plane for the surface temperature [K].

    Returns:
      A dictionary containing the Planck source at the cell center
      (`planck_src`), the top cell boundary (`planck_src_top`), and the bottom
      cell boundary (`planck_src_bottom`).
    """

  @property
  @abc.abstractmethod
  def n_gpt_lw(self) -> int:
    """The number of g-points in the longwave bands."""

  @property
  @abc.abstractmethod
  def n_gpt_sw(self) -> int:
    """The number of g-points in the shortwave bands."""

  @property
  @abc.abstractmethod
  def solar_fraction_by_gpt(self) -> jax.Array:
    """Mapping from g-point to the fraction of total solar radiation."""

  def _exchange_halos(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      f: ScalarField,
  ) -> ScalarField:
    """Exchanges halos, preserving the boundary values along the vertical."""
    axes = ('x', 'y', 'z')
    # Build boundary conditions matching BoundaryConditionsSpec.
    neumann_bc = (halo_exchange.BCType.NEUMANN, 0.0)
    # For the vertical dimension, use Dirichlet BCs to preserve boundary values.
    lower_vals: list[jax.Array] = [
        jnp.take(f, i, axis=self._g_dim) for i in range(self._halos)
    ]
    upper_vals: list[jax.Array] = [
        jnp.take(f, f.shape[self._g_dim] - 1 - i, axis=self._g_dim)
        for i in range(self._halos)
    ]
    upper_vals.reverse()
    dirichlet_low = (halo_exchange.BCType.DIRICHLET, lower_vals)
    dirichlet_high = (halo_exchange.BCType.DIRICHLET, upper_vals)

    bc_list: list[tuple | None] = [(neumann_bc, neumann_bc)] * 3  # pylint: disable=g-bare-generic
    bc_list[self._g_dim] = (dirichlet_low, dirichlet_high)
    bc_spec = tuple(bc_list)

    return halo_exchange.inplace_halo_exchange(
        f,
        axes,
        mesh,
        grid_params,
        periodic_dims=[False, False, False],
        boundary_conditions=bc_spec,
        halo_width=self._halos,
    )

  def _reconstruct_face_values(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      f: ScalarField,
  ) -> tuple[ScalarField, ScalarField]:
    """Reconstructs the face values using a high-order scheme.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      f: The cell-center values that will be interpolated.

    Returns:
      A tuple with the reconstructed temperature at the bottom and top face,
      respectively.
    """
    dim = ('x', 'y', 'z')[self._g_dim]
    f_neg, f_pos = interpolation.weno(
        f,
        axis=dim,
        k=self._face_interp_scheme_order,
        grid_params=grid_params,
        kernel_type='conv',
    )
    f_bottom = self._exchange_halos(
        mesh,
        grid_params,
        0.5 * (f_neg + f_pos),
    )

    # Shift down to obtain the top cell face values and pad the top outermost
    # halo layer with a copy of the adjacent inner layer.
    f_top = jnp.roll(f_bottom, 1, axis=self._g_dim)
    # Get the outermost valid top layer (second from last along g_dim).
    slices_outer: list[int | slice] = [slice(None)] * f_top.ndim  # pyrefly: ignore[code]
    slices_outer[self._g_dim] = -2
    outermost_valid = f_top[tuple(slices_outer)]
    # Update the last halo layer along the vertical.
    slices_last: list[int | slice] = [slice(None)] * f_top.ndim  # pyrefly: ignore[code]
    slices_last[self._g_dim] = -1
    f_top = f_top.at[tuple(slices_last)].set(outermost_valid)
    return f_bottom, f_top

  def combine_optical_properties(
      self,
      optical_props_1: ScalarFieldMap,
      optical_props_2: ScalarFieldMap,
  ) -> ScalarFieldMap:
    """Combines the optical properties from two separate parameterizations."""
    tau = optical_props_1['optical_depth'] + optical_props_2['optical_depth']

    ssa_unnormalized = (
        optical_props_1['optical_depth'] * optical_props_1['ssa']
        + optical_props_2['optical_depth'] * optical_props_2['ssa']
    )

    def divide(x: jax.Array, y: jax.Array) -> jax.Array:
      return x / jnp.maximum(y, self._EPSILON)

    g = divide(
        optical_props_1['optical_depth']
        * optical_props_1['ssa']
        * optical_props_1['asymmetry_factor']
        + optical_props_2['optical_depth']
        * optical_props_2['ssa']
        * optical_props_2['asymmetry_factor'],
        ssa_unnormalized,
    )
    return {
        'optical_depth': tau,
        'ssa': divide(ssa_unnormalized, tau),
        'asymmetry_factor': g,
    }
