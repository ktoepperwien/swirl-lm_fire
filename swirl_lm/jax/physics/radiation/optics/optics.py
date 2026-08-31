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

"""Implementations of `OpticsScheme`s and a factory method (JAX).

JAX port of `swirl_lm.physics.radiation.optics.optics`.
"""


from typing import Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.jax.physics.radiation.optics import cloud_optics
from swirl_lm.jax.physics.radiation.optics import gas_optics
from swirl_lm.jax.physics.radiation.optics import lookup_cloud_optics as cloud_lookup_lib
from swirl_lm.jax.physics.radiation.optics import lookup_gas_optics_base
from swirl_lm.jax.physics.radiation.optics import lookup_gas_optics_longwave
from swirl_lm.jax.physics.radiation.optics import lookup_gas_optics_shortwave
from swirl_lm.jax.physics.radiation.optics import lookup_volume_mixing_ratio
from swirl_lm.jax.physics.radiation.optics import optics_base
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types
from swirl_lm.physics.radiation.config import radiative_transfer_pb2
from swirl_lm.physics.radiation.optics import constants

AbstractLookupGasOptics = lookup_gas_optics_base.AbstractLookupGasOptics
LookupCloudOptics = cloud_lookup_lib.LookupCloudOptics
LookupGasOpticsLongwave = lookup_gas_optics_longwave.LookupGasOpticsLongwave
LookupGasOpticsShortwave = lookup_gas_optics_shortwave.LookupGasOpticsShortwave
LookupVolumeMixingRatio = lookup_volume_mixing_ratio.LookupVolumeMixingRatio
ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


class RRTMOptics(optics_base.OpticsScheme):
  """The Rapid Radiative Transfer Model (RRTM) optics scheme implementation."""

  def __init__(
      self,
      vmr_lib: LookupVolumeMixingRatio,
      params: radiative_transfer_pb2.OpticsParameters,
      g_dim: int,
      halos: int,
  ):
    super().__init__(params, g_dim, halos)
    rrtm_params = params.rrtm_optics
    self.vmr_lib: LookupVolumeMixingRatio = vmr_lib
    self.cloud_optics_lw: LookupCloudOptics = LookupCloudOptics.from_nc_file(
        rrtm_params.cloud_longwave_nc_filepath
    )
    self.cloud_optics_sw: LookupCloudOptics = LookupCloudOptics.from_nc_file(
        rrtm_params.cloud_shortwave_nc_filepath
    )
    self.gas_optics_lw: LookupGasOpticsLongwave = (
        LookupGasOpticsLongwave.from_nc_file(rrtm_params.longwave_nc_filepath)
    )
    self.gas_optics_sw: LookupGasOpticsShortwave = (
        LookupGasOpticsShortwave.from_nc_file(rrtm_params.shortwave_nc_filepath)
    )

  def _compute_optical_depth(
      self,
      is_lw: bool,
      igpt: jax.Array,
      molecules: ScalarField,
      temperature: ScalarField,
      pressure: ScalarField,
      vmr_fields: Optional[dict[int, jax.Array]],
  ) -> ScalarField:
    """Computes the optical depth for gas absorption."""
    lookup_gas_optics = self.gas_optics_lw if is_lw else self.gas_optics_sw
    return gas_optics.compute_minor_optical_depth(
        lookup_gas_optics,
        self.vmr_lib,
        molecules,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    ) + gas_optics.compute_major_optical_depth(
        lookup_gas_optics,
        self.vmr_lib,
        molecules,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    )

  def _compute_rayleigh_scattering(
      self,
      igpt: jax.Array,
      molecules: ScalarField,
      temperature: ScalarField,
      pressure: ScalarField,
      vmr_fields: Optional[dict[int, jax.Array]],
  ) -> ScalarField:
    """Computes the Rayleigh scattering optical depth."""
    return gas_optics.compute_rayleigh_optical_depth(
        self.gas_optics_sw,
        self.vmr_lib,
        molecules,
        temperature,
        pressure,
        igpt,
        vmr_fields,
    )

  def _compute_planck_fraction(
      self,
      igpt: jax.Array,
      pressure: ScalarField,
      temperature: ScalarField,
      vmr_fields: Optional[dict[int, jax.Array]] = None,
  ) -> ScalarField:
    """Computes the Planck fraction."""
    return gas_optics.compute_planck_fraction(
        self.gas_optics_lw,
        self.vmr_lib,
        pressure,
        temperature,
        igpt,
        vmr_fields,
    )

  def _compute_planck_source(
      self,
      igpt: jax.Array,
      planck_fraction: ScalarField,
      temperature: ScalarField,
  ) -> ScalarField:
    """Computes the Planck source."""
    return gas_optics.compute_planck_sources(
        self.gas_optics_lw,
        planck_fraction,
        temperature,
        igpt,
    )

  def _compute_cloud_properties(
      self,
      ibnd: jax.Array,
      is_lw: bool,
      r_eff_liq: ScalarField,
      cloud_path_liq: ScalarField,
      r_eff_ice: ScalarField,
      cloud_path_ice: ScalarField,
  ) -> dict[str, jax.Array]:
    """Computes the cloud optical properties."""
    cloud_lookup = self.cloud_optics_lw if is_lw else self.cloud_optics_sw
    return cloud_optics.compute_optical_properties(
        cloud_lookup,
        cloud_path_liq,
        cloud_path_ice,
        r_eff_liq,
        r_eff_ice,
        ibnd=ibnd,
    )

  def _apply_delta_scaling_for_cloud(
      self,
      cloud_optical_props: ScalarFieldMap,
  ) -> ScalarFieldMap:
    """Delta-scales optical properties for shortwave bands."""
    tau = cloud_optical_props['optical_depth']
    ssa = cloud_optical_props['ssa']
    g = cloud_optical_props['asymmetry_factor']

    wf = ssa * g**2
    cloud_tau = (1.0 - wf) * tau
    cloud_ssa = jnp.where(
        1.0 - wf < self._EPSILON,
        0.0,
        (ssa - wf) / jnp.maximum(self._EPSILON, 1.0 - wf),
    )
    f = g**2
    cloud_asy = jnp.where(
        1.0 - f < self._EPSILON,
        0.0,
        (g - f) / jnp.maximum(self._EPSILON, 1.0 - f),
    )
    return {
        'optical_depth': cloud_tau,
        'ssa': cloud_ssa,
        'asymmetry_factor': cloud_asy,
    }

  def _combine_gas_and_cloud_properties(
      self,
      igpt: jax.Array,
      optical_props: ScalarFieldMap,
      is_lw: bool,
      radius_eff_liq: Optional[ScalarField] = None,
      cloud_path_liq: Optional[ScalarField] = None,
      radius_eff_ice: Optional[ScalarField] = None,
      cloud_path_ice: Optional[ScalarField] = None,
  ) -> ScalarFieldMap:
    """Combines the gas optical properties with the cloud optical properties."""
    gas_lookup = self.gas_optics_lw if is_lw else self.gas_optics_sw
    cloud_states = [
        radius_eff_liq,
        cloud_path_liq,
        radius_eff_ice,
        cloud_path_ice,
    ]
    cloud_states = [
        x if x is not None else jnp.zeros_like(optical_props['ssa'])
        for x in cloud_states
    ]
    ibnd = gas_lookup.g_point_to_bnd[igpt]  # pylint: disable=attribute-error

    cloud_optical_props = self._compute_cloud_properties(
        ibnd, is_lw, *cloud_states
    )
    if not is_lw:
      cloud_optical_props = self._apply_delta_scaling_for_cloud(
          cloud_optical_props
      )
    return self.combine_optical_properties(optical_props, cloud_optical_props)

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

    Uses the RRTM optics scheme to compute the longwave optical depth, albedo,
    and asymmetry factor.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules / m**2].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path [kg/m**2].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path [kg/m**2].

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The longwave optical depth.
        'ssa': The longwave single-scattering albedo.
        'asymmetry_factor': The longwave asymmetry factor.
    """
    optical_depth_lw = self._compute_optical_depth(
        True,
        igpt,
        molecules,
        temperature,
        pressure,
        vmr_fields,
    )
    zeros = jnp.zeros_like(optical_depth_lw)
    optical_props = {
        'optical_depth': optical_depth_lw,
        'ssa': zeros,
        'asymmetry_factor': zeros,
    }
    if cloud_path_liq is not None or cloud_path_ice is not None:
      return self._combine_gas_and_cloud_properties(
          igpt,
          optical_props,
          is_lw=True,
          radius_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=cloud_path_liq,
          radius_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=cloud_path_ice,
      )
    return optical_props

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

    Uses the RRTM optics scheme to compute the shortwave optical depth, albedo,
    and asymmetry factor.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules / m**2].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path [kg/m**2].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path [kg/m**2].

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The shortwave optical depth.
        'ssa': The shortwave single-scattering albedo.
        'asymmetry_factor': The shortwave asymmetry factor.
    """
    optical_depth_sw = self._compute_optical_depth(
        False,
        igpt,
        molecules,
        temperature,
        pressure,
        vmr_fields,
    )
    rayleigh_scattering = self._compute_rayleigh_scattering(
        igpt,
        molecules,
        temperature,
        pressure,
        vmr_fields,
    )
    optical_depth_sw = optical_depth_sw + rayleigh_scattering
    ssa = jnp.where(
        optical_depth_sw == 0, 0.0, rayleigh_scattering / optical_depth_sw
    )
    gas_optical_props = {
        'optical_depth': optical_depth_sw,
        'ssa': ssa,
        'asymmetry_factor': jnp.zeros_like(ssa),
    }
    if cloud_path_liq is not None or cloud_path_ice is not None:
      return self._combine_gas_and_cloud_properties(
          igpt,
          gas_optical_props,
          is_lw=False,
          radius_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=cloud_path_liq,
          radius_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=cloud_path_ice,
      )
    return gas_optical_props

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
    """Computes the monochromatic Planck sources given the atmospheric state.

    This requires interpolating the temperature at cell faces using a high-order
    scheme.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      sfc_temperature: The optional surface temperature [K].

    Returns:
      A dictionary containing the Planck source at the cell center
      (`planck_src`), the top cell boundary (`planck_src_top`), the bottom cell
      boundary (`planck_src_bottom`) and, if a `sfc_temperature` argument was
      provided, the surface cell boundary (`planck_src_sfc`).
    """
    temperature_bottom, temperature_top = self._reconstruct_face_values(
        mesh, grid_params, temperature
    )

    planck_fraction = self._compute_planck_fraction(
        igpt,
        pressure,
        temperature,
        vmr_fields,
    )

    planck_src_fn = lambda pf, t: self._compute_planck_source(igpt, pf, t)

    planck_srcs: dict[str, ScalarField] = {
        'planck_src': planck_src_fn(planck_fraction, temperature),
        'planck_src_top': planck_src_fn(planck_fraction, temperature_top),
        'planck_src_bottom': planck_src_fn(planck_fraction, temperature_bottom),
    }

    if sfc_temperature is not None:
      g_axis = ('x', 'y', 'z')[self._g_dim]
      planck_fraction_0 = common_ops.get_face(
          planck_fraction,
          g_axis,
          face=0,
          index=self._halos,
          grid_params=grid_params,
      )
      planck_src_sfc = planck_src_fn(planck_fraction_0, sfc_temperature)
      # Only allow the first computational layer of cores to have a nonzero
      # surface Planck source. This is determined at runtime by whether the
      # current device is in the first position along the vertical axis.
      planck_srcs['planck_src_sfc'] = planck_src_sfc

    return planck_srcs

  @property
  def n_gpt_lw(self) -> int:
    """The number of g-points in the longwave bands."""
    if self.gas_optics_lw is None:
      raise ValueError('Longwave gas optics lookup table not initialized.')
    return self.gas_optics_lw.n_gpt

  @property
  def n_gpt_sw(self) -> int:
    """The number of g-points in the shortwave bands."""
    if self.gas_optics_sw is None:
      raise ValueError('Shortwave gas optics lookup table not initialized.')
    return self.gas_optics_sw.n_gpt

  @property
  def solar_fraction_by_gpt(self) -> jax.Array:
    """Mapping from g-point to the fraction of total solar radiation."""
    if self.gas_optics_sw is None:
      raise ValueError('Shortwave gas optics lookup table not initialized.')
    return self.gas_optics_sw.solar_src_scaled


class GrayAtmosphereOptics(optics_base.OpticsScheme):
  """Implementation of the gray atmosphere optics scheme."""

  def __init__(
      self,
      params: radiative_transfer_pb2.OpticsParameters,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      g_dim: int,
      halos: int,
  ):
    super().__init__(params, g_dim, halos)
    self._p0 = params.gray_atmosphere_optics.p0
    self._alpha = params.gray_atmosphere_optics.alpha
    self._d0_lw = params.gray_atmosphere_optics.d0_lw
    self._d0_sw = params.gray_atmosphere_optics.d0_sw
    self._kernel_op = kernel_op
    self._dim_str = ('x', 'y', 'z')[g_dim]
    kernel_name = ('kDx', 'kDy', 'kDz')[g_dim]
    self._grad_central_name = kernel_name

  def compute_lw_optical_properties(
      self,
      pressure: ScalarField,
      *args,
      **kwargs,
  ) -> ScalarFieldMap:
    """Computes longwave optical properties based on pressure and lapse rate.

    See Schneider 2004, J. Atmos. Sci. (2004) 61 (12): 1317-1340.
    DOI: https://doi.org/10.1175/1520-0469(2004)061<1317:TTATTS>2.0.CO;2
    To obtain the local optical depth of the layer, the expression for
    cumulative optical depth (from the top of the atmosphere to an arbitrary
    pressure level) was differentiated with respect to the pressure and
    multiplied by the pressure difference across the grid cell.

    Args:
      pressure: The pressure field [Pa].
      *args: Miscellaneous inherited arguments.
      **kwargs: Miscellaneous inherited keyword arguments.

    Returns:
      A dictionary containing the optical depth (`optical_depth`), the single-
      scattering albedo (`ssa`), and the asymmetry factor (`asymmetry_factor`)
      for longwave radiation.
    """
    dp = (
        self._kernel_op.apply_kernel_op(
            pressure, self._grad_central_name, self._dim_str
        )
        / 2.0
    )
    tau = jnp.abs(
        self._alpha
        * self._d0_lw
        * jnp.power(pressure / self._p0, self._alpha)
        / pressure
        * dp
    )
    return {
        'optical_depth': tau,
        'ssa': jnp.zeros_like(pressure),
        'asymmetry_factor': jnp.zeros_like(pressure),
    }

  def compute_sw_optical_properties(
      self,
      pressure: ScalarField,
      *args,
      **kwargs,
  ) -> ScalarFieldMap:
    """Computes the shortwave optical properties of a gray atmosphere.

    See O'Gorman 2008, Journal of Climate Vol 21, Page(s): 3815-3832.
    DOI: https://doi.org/10.1175/2007JCLI2065.1. In particular, the cumulative
    optical depth expression shown in equation 3 inside the exponential is
    differentiated with respect to pressure and scaled by the pressure
    difference across the grid cell.

    Args:
      pressure: The pressure field [Pa].
      *args: Miscellaneous inherited arguments.
      **kwargs: Miscellaneous inherited keyword arguments.

    Returns:
      A dictionary containing the optical depth (`optical_depth`), the single-
      scattering albedo (`ssa`), and the asymmetry factor (`asymmetry_factor`)
      for shortwave radiation.
    """
    dp = (
        self._kernel_op.apply_kernel_op(
            pressure, self._grad_central_name, self._dim_str
        )
        / 2.0
    )
    optical_depth = jnp.abs(
        2.0 * self._d0_sw * (pressure / self._p0) * (dp / self._p0)
    )
    return {
        'optical_depth': optical_depth,
        'ssa': jnp.zeros_like(optical_depth),
        'asymmetry_factor': jnp.zeros_like(optical_depth),
    }

  def compute_planck_sources(  # pyrefly: ignore[bad-override]
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      pressure: ScalarField,
      temperature: ScalarField,
      *args,
      sfc_temperature: Optional[ScalarField] = None,
  ) -> ScalarFieldMap:
    """Computes the Planck sources used in the longwave problem.

    The computation is based on Stefan-Boltzmann's law, which states that the
    thermal radiation emitted from a black body is directly proportional to the
    4-th power of its absolute temperature.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      *args: Miscellaneous inherited arguments.
      sfc_temperature: The optional surface temperature [K].

    Returns:
      A dictionary containing the Planck source at the cell center
      (`planck_src`), the top cell boundary (`planck_src_top`), and the bottom
      cell boundary (`planck_src_bottom`).
    """
    del pressure

    def src_fn(t: jax.Array) -> jax.Array:
      return constants.STEFAN_BOLTZMANN * t**4 / np.pi

    temperature_bottom, temperature_top = self._reconstruct_face_values(
        mesh,
        grid_params,
        temperature,
    )

    planck_srcs: dict[str, ScalarField] = {
        'planck_src': src_fn(temperature),
        'planck_src_top': src_fn(temperature_top),
        'planck_src_bottom': src_fn(temperature_bottom),
    }
    if sfc_temperature is not None:
      planck_srcs['planck_src_sfc'] = src_fn(sfc_temperature)
    return planck_srcs

  @property
  def n_gpt_lw(self) -> int:
    """The number of g-points in the longwave bands."""
    return 1

  @property
  def n_gpt_sw(self) -> int:
    """The number of g-points in the shortwave bands."""
    return 1

  @property
  def solar_fraction_by_gpt(self) -> jax.Array:
    """Mapping from g-point to the fraction of total solar radiation."""
    return jnp.array([1.0], dtype=jnp.float32)


def optics_factory(
    params: radiative_transfer_pb2.OpticsParameters,
    kernel_op: get_kernel_fn.ApplyKernelOp,
    g_dim: int,
    halos: int,
    vmr_lib: Optional[LookupVolumeMixingRatio] = None,
) -> optics_base.OpticsScheme:
  """Constructs an instance of `OpticsScheme`.

  Args:
    params: The optics parameters.
    kernel_op: An object holding a library of kernel operations. Used only by
      `GrayAtmosphereOptics` for gradient calculations.
    g_dim: The vertical dimension.
    halos: The number of halo layers.
    vmr_lib: An instance of `LookupVolumeMixingRatio` containing gas
      concentrations.

  Returns:
    An instance of `OpticsScheme`.
  """
  if params.HasField('rrtm_optics'):
    assert vmr_lib is not None, '`vmr_lib` is required for `RRTMOptics`.'
    return RRTMOptics(vmr_lib, params, g_dim=g_dim, halos=halos)
  elif params.HasField('gray_atmosphere_optics'):
    return GrayAtmosphereOptics(params, kernel_op, g_dim=g_dim, halos=halos)
  else:
    raise ValueError('Unsupported optics scheme.')
