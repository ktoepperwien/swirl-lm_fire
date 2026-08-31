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

"""A library for solving the two-stream radiative transfer equation (JAX).

JAX port of `swirl_lm.physics.radiation.rte.two_stream`.
"""


import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.jax.physics.radiation.optics import atmospheric_state
from swirl_lm.jax.physics.radiation.optics import lookup_gas_optics_base
from swirl_lm.jax.physics.radiation.optics import optics
from swirl_lm.jax.physics.radiation.rte import monochromatic_two_stream
from swirl_lm.jax.physics.radiation.rte import rte_utils as utils
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types
from swirl_lm.physics import constants
from swirl_lm.physics.radiation.config import radiative_transfer_pb2

AbstractLookupGasOptics = lookup_gas_optics_base.AbstractLookupGasOptics
AtmosphericState = atmospheric_state.AtmosphericState
ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

PRIMARY_GRID_KEY = utils.PRIMARY_GRID_KEY
EXTENDED_GRID_KEY = utils.EXTENDED_GRID_KEY


class TwoStreamSolver:
  """A library for solving the two-stream radiative transfer equation.

  Attributes:
    atmospheric_state: An instance of `AtmosphericState` containing volume
      mixing ratio profiles for prevalent atmospheric gases and flux boundary
      conditions at particular times and geographic locations.
  """

  def __init__(
      self,
      radiation_params: radiative_transfer_pb2.RadiativeTransfer,
      grid_params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      g_dim: int,
  ):
    self.atmospheric_state = AtmosphericState.from_proto(
        radiation_params.atmospheric_state
    )
    self._g_dim = g_dim
    self._halos = grid_params.halo_width
    self._optics_lib = optics.optics_factory(
        radiation_params.optics,
        kernel_op,
        g_dim,
        self._halos,
        self.atmospheric_state.vmr,
    )
    self._monochrom_solver = (
        monochromatic_two_stream.MonochromaticTwoStreamSolver(
            grid_params,
            kernel_op,
            g_dim,
        )
    )
    self._rte_utils = utils.RTEUtils(grid_params)

    # Operators used when computing heating rate from fluxes.
    self._kernel_op = kernel_op
    dim_str = ('x', 'y', 'z')[g_dim]
    grad_kernel = ('kDx', 'kDy', 'kDz')[g_dim]
    fwd_kernel = ('kdx+', 'kdy+', 'kdz+')[g_dim]
    self._grad_central_kernel = grad_kernel
    self._grad_forward_kernel = fwd_kernel
    self._dim_str = dim_str

    # Longwave parameters.
    self._top_flux_down_lw = self.atmospheric_state.toa_flux_lw
    self._sfc_emissivity_lw = self.atmospheric_state.sfc_emis

    # Shortwave parameters.
    self._sfc_albedo = self.atmospheric_state.sfc_alb
    self._zenith = self.atmospheric_state.zenith
    self._total_solar_irrad = self.atmospheric_state.irrad
    self._solar_fraction_by_gpt = self._optics_lib.solar_fraction_by_gpt
    self._flux_keys = ['flux_up', 'flux_down', 'flux_net']

  def _compute_local_properties_lw(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      pressure: ScalarField,
      temperature: ScalarField,
      molecules: ScalarField,
      igpt: jax.Array,
      vmr_fields: dict[int, ScalarField] | None = None,
      sfc_temperature: ScalarField | float | None = None,
      cloud_r_eff_liq: ScalarField | None = None,
      cloud_path_liq: ScalarField | None = None,
      cloud_r_eff_ice: ScalarField | None = None,
      cloud_path_ice: ScalarField | None = None,
  ) -> dict[str, ScalarField]:
    """Computes local optical properties for longwave radiative transfer."""
    if isinstance(sfc_temperature, float):
      # Create a plane for the surface temperature representation.
      sfc_plane = common_ops.slice_field(pressure, self._g_dim, 0, size=1)
      sfc_temperature = sfc_temperature * jnp.ones_like(sfc_plane)
    lw_optical_props = dict(
        self._optics_lib.compute_lw_optical_properties(
            pressure,
            temperature,
            molecules,
            igpt,
            vmr_fields=vmr_fields,
            cloud_r_eff_liq=cloud_r_eff_liq,
            cloud_path_liq=cloud_path_liq,
            cloud_r_eff_ice=cloud_r_eff_ice,
            cloud_path_ice=cloud_path_ice,
        )
    )
    planck_srcs = dict(
        self._optics_lib.compute_planck_sources(
            mesh,
            grid_params,
            pressure,
            temperature,
            igpt,
            vmr_fields,
            sfc_temperature=sfc_temperature,
        )
    )
    sfc_src = planck_srcs.get(
        'planck_src_sfc',
        common_ops.slice_field(
            planck_srcs['planck_src_bottom'],
            self._g_dim,
            self._halos,
            size=1,
        ),
    )
    combined_srcs = self._monochrom_solver.lw_combine_sources(planck_srcs)
    lw_optical_props['level_src_bottom'] = combined_srcs['planck_src_bottom']
    lw_optical_props['level_src_top'] = combined_srcs['planck_src_top']
    src_and_props = dict(
        self._monochrom_solver.lw_cell_source_and_properties(**lw_optical_props)
    )
    src_and_props['sfc_src'] = sfc_src
    return src_and_props

  def _reindex_vmr_fields(
      self,
      vmr_fields: dict[str, ScalarField],
      gas_optics_lib: AbstractLookupGasOptics,
  ) -> dict[int, ScalarField]:
    """Converts the chemical formulas of the gas species to RRTM indices."""
    return {gas_optics_lib.idx_gases[k]: v for k, v in vmr_fields.items()}

  def solve_lw(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      replicas: np.ndarray,
      pressure: ScalarField,
      temperature: ScalarField,
      molecules: ScalarField,
      vmr_fields: dict[str, ScalarField] | None = None,
      sfc_temperature: ScalarField | float | None = None,
      cloud_r_eff_liq: ScalarField | None = None,
      cloud_path_liq: ScalarField | None = None,
      cloud_r_eff_ice: ScalarField | None = None,
      cloud_path_ice: ScalarField | None = None,
  ) -> dict[str, ScalarField]:
    """Solves two-stream radiative transfer equation over the longwave spectrum.

    Local optical properties like optical depth, single-scattering albedo, and
    asymmetry factor are computed using an optics library and transformed to
    two-stream approximations of reflectance and transmittance. The sources of
    longwave radiation are the Planck sources, which are a function only of
    temperature. To obtain the cell-centered directional Planck sources, the
    sources are first computed at the cell boundaries and the net source
    emanating from the grid cell is determined. Each spectral interval,
    represented by a g-point, is a separate radiative transfer problem, and can
    be computed in parallel. Finally, the independently solved fluxes are summed
    over the full spectrum to yield the final upwelling and downwelling fluxes.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      replicas: The mapping from the core coordinate to the local replica id.
      pressure: The pressure field [Pa].
      temperature: The temperature field [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m**2].
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by the chemical formula.
      sfc_temperature: The optional surface temperature represented as either a
        3D field having a single vertical dimension or as a scalar [K].
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m**2].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m**2].

    Returns:
      A dictionary with the following entries (in units of W/m**2):
      `flux_up`: The upwelling longwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling longwave radiative flux at face i - 1/2.
      `flux_net`: The net longwave radiative flux at face i - 1/2.
    """
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    indexed_vmr_fields: dict[int, ScalarField] | None = None  # pylint: disable=unused-variable
    if vmr_fields is not None and self._optics_lib.gas_optics_lw is not None:
      gas_optics_lib = self._optics_lib.gas_optics_lw
      indexed_vmr_fields = self._reindex_vmr_fields(vmr_fields, gas_optics_lib)
    else:
      indexed_vmr_fields = vmr_fields  # pyrefly: ignore[code]

    def step_fn(igpt: jax.Array, cumulative_flux: dict[str, ScalarField]):
      optical_props_2stream = self._compute_local_properties_lw(
          mesh,
          grid_params,
          pressure,
          temperature,
          molecules,
          igpt,
          indexed_vmr_fields,
          sfc_temperature,
          cloud_r_eff_liq,
          cloud_path_liq,
          cloud_r_eff_ice,
          cloud_path_ice,
      )

      # Boundary conditions.
      sfc_src = optical_props_2stream['sfc_src']
      top_flux_down = self._top_flux_down_lw * jnp.ones_like(sfc_src)
      sfc_emissivity = self._sfc_emissivity_lw * jnp.ones_like(sfc_src)
      fluxes = self._monochrom_solver.lw_transport(
          mesh,
          grid_params,
          replicas,
          top_flux_down=top_flux_down,
          sfc_emissivity=sfc_emissivity,
          **optical_props_2stream,
      )
      updated = {k: cumulative_flux[k] + fluxes[k] for k in cumulative_flux}
      return updated

    lw_fluxes: dict[str, ScalarField] = {
        k: jnp.zeros_like(pressure) for k in self._flux_keys
    }
    for i in range(self._optics_lib.n_gpt_lw):
      lw_fluxes = step_fn(jnp.int32(i), lw_fluxes)
    return lw_fluxes

  def solve_sw(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      replicas: np.ndarray,
      pressure: ScalarField,
      temperature: ScalarField,
      molecules: ScalarField,
      vmr_fields: dict[str, ScalarField] | None = None,
      cloud_r_eff_liq: ScalarField | None = None,
      cloud_path_liq: ScalarField | None = None,
      cloud_r_eff_ice: ScalarField | None = None,
      cloud_path_ice: ScalarField | None = None,
  ) -> dict[str, ScalarField]:
    """Solves the two-stream radiative transfer equation for shortwave.

    Local optical properties like optical depth, single-scattering albedo, and
    asymmetry factor are computed using an optics library and transformed to
    two-stream approximations of reflectance and transmittance. The sources of
    shortwave radiation are determined by the diffuse propagation of direct
    solar radiation through the layered atmosphere. Each spectral interval,
    represented by a g-point, is a separate radiative transfer problem, and can
    be computed in parallel. Finally, the independently solved fluxes are summed
    over the full spectrum to yield the final upwelling and downwelling fluxes.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      replicas: The mapping from the core coordinate to the local replica id.
      pressure: The pressure field [Pa].
      temperature: The temperature field [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m**2].
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m**2].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m**2].

    Returns:
      A dictionary with the following entries (in units of W/m**2):
      `flux_up`: The upwelling shortwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling shortwave radiative flux at face i - 1/2.
      `flux_net`: The net shortwave radiative flux at face i - 1/2.
    """
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    indexed_vmr_fields: dict[int, ScalarField] | None = None  # pylint: disable=unused-variable
    if vmr_fields is not None and self._optics_lib.gas_optics_sw is not None:
      gas_optics_lib = self._optics_lib.gas_optics_sw
      indexed_vmr_fields = self._reindex_vmr_fields(vmr_fields, gas_optics_lib)
    else:
      indexed_vmr_fields = vmr_fields  # pyrefly: ignore[code]

    fluxes: dict[str, ScalarField] = {
        k: jnp.zeros_like(pressure) for k in self._flux_keys
    }
    if self._zenith >= 0.5 * np.pi:
      return fluxes

    def step_fn(igpt: jax.Array, partial_fluxes: dict[str, ScalarField]):
      sw_optical_props = self._optics_lib.compute_sw_optical_properties(
          pressure,
          temperature,
          molecules,
          igpt,
          vmr_fields=indexed_vmr_fields,
          cloud_r_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=cloud_path_liq,
          cloud_r_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=cloud_path_ice,
      )
      optical_props_2stream = self._monochrom_solver.sw_cell_properties(
          zenith=self._zenith,
          **sw_optical_props,
      )
      sfc_albedo = self._sfc_albedo * jnp.ones_like(
          common_ops.slice_field(
              sw_optical_props['optical_depth'], self._g_dim, 0, size=1
          )
      )
      # Monochromatic top of atmosphere flux.
      solar_flux = self._total_solar_irrad * self._solar_fraction_by_gpt[igpt]
      toa_flux = solar_flux * jnp.ones_like(sfc_albedo)

      sources_2stream_full_grid = self._monochrom_solver.sw_cell_source(
          mesh,
          grid_params,
          replicas,
          t_dir=optical_props_2stream['t_dir'],
          r_dir=optical_props_2stream['r_dir'],
          optical_depth=sw_optical_props['optical_depth'],
          toa_flux=toa_flux,
          sfc_albedo_direct=sfc_albedo,
          zenith=self._zenith,
      )

      # Extract out primary grid sources.
      sources_2stream = sources_2stream_full_grid[PRIMARY_GRID_KEY]

      sw_fluxes = self._monochrom_solver.sw_transport(
          mesh,
          grid_params,
          replicas,
          t_diff=optical_props_2stream['t_diff'],
          r_diff=optical_props_2stream['r_diff'],
          src_up=sources_2stream['src_up'],
          src_down=sources_2stream['src_down'],
          sfc_src=sources_2stream['sfc_src'],
          sfc_albedo=sfc_albedo,
          flux_down_dir=sources_2stream['flux_down_dir'],
      )
      updated = {k: partial_fluxes[k] + sw_fluxes[k] for k in partial_fluxes}
      return updated

    for i in range(self._optics_lib.n_gpt_sw):
      fluxes = step_fn(jnp.int32(i), fluxes)
    return fluxes

  def compute_heating_rate(
      self,
      flux_net: ScalarField,
      pressure: ScalarField,
  ) -> ScalarField:
    """Computes cell-center heating rate from pressure and net radiative flux.

    The net radiative flux corresponds to the bottom cell face. The difference
    of the net flux at the top face and that at the bottom face gives the total
    net flux out of the grid cell. Using the pressure difference across the grid
    cell, the net flux can be converted to a heating rate, in K/s.

    Args:
      flux_net: The net flux at the bottom face [W/m**2].
      pressure: The pressure field [Pa].

    Returns:
      The heating rate of the grid cell [K/s].
    """
    # Pressure difference across the atmospheric grid cell.
    dp = (
        self._kernel_op.apply_kernel_op(
            pressure, self._grad_central_kernel, self._dim_str
        )
        / 2.0
    )
    # Net flux difference across the grid cell.
    dflux = self._kernel_op.apply_kernel_op(
        flux_net, self._grad_forward_kernel, self._dim_str
    )
    return constants.G * dflux / dp / constants.CP  # pyrefly: ignore[code]
