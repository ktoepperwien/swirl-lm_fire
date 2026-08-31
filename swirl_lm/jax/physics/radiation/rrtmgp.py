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

"""Implementation of a radiative transfer solver (JAX).

JAX port of `swirl_lm.physics.radiation.rrtmgp`.
"""


from typing import Any

import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.physics.atmosphere import microphysics_one_moment
from swirl_lm.jax.physics.radiation.rte import two_stream
from swirl_lm.jax.physics.thermodynamics import water
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import stretched_grid_util
from swirl_lm.jax.utility import types
from swirl_lm.physics import constants
from swirl_lm.physics.radiation import rrtmgp_common

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

PRIMARY_GRID_KEY = two_stream.PRIMARY_GRID_KEY


class RRTMGP:
  """Rapid Radiative Transfer Model for General Circulation Models (RRTMGP)."""

  def __init__(
      self,
      config: parameters_lib.SwirlLMParameters,
  ):
    self._kernel_op = config.kernel_op
    self._kernel_op.add_kernel({
        'shift_up': ([1.0, 0.0, 0.0], 1),
        'shift_dn': ([0.0, 0.0, 1.0], 1),
    })
    self._config = config
    # A thermodynamics manager that handles moisture related physics.
    self._water = water.Water(config)
    # The vertical dimension.
    self._g_dim = config.g_dim
    # The number of ghost points on a side of the subgrid.
    self._halos = config.halo_width
    # The vertical grid spacing used in computing the local water path for an
    # atmospheric grid cell.
    self._dh = config.grid_spacings[self._g_dim]  # pyrefly: ignore[type-annotation]
    # Whether stretched grid is used in each dimension.
    self._use_stretched_grid = config.use_stretched_grid
    # The two-stream radiative transfer solver.
    self._two_stream_solver = two_stream.TwoStreamSolver(
        config.radiative_transfer,  # pyrefly: ignore[code]
        config.grid_params,
        self._kernel_op,
        self._g_dim,  # pyrefly: ignore[code]
    )
    # Data library containing atmospheric gas concentrations.
    self._atmospheric_state = self._two_stream_solver.atmospheric_state
    # Library for 1-moment microphysics.
    self._microphysics_lib = microphysics_one_moment.Adapter(
        config, self._water
    )
    self._vertical_coord_name = ('xx', 'yy', 'zz')[self._g_dim]  # pyrefly: ignore[code]

  def _compute_cloud_path(
      self,
      rho: ScalarField,
      q_c: ScalarField,
      additional_states: ScalarFieldMap,
  ) -> ScalarField:
    """Computes the cloud water/ice path in an atmospheric grid cell."""
    if self._use_stretched_grid[self._g_dim]:  # pyrefly: ignore[code]
      h = additional_states[stretched_grid_util.h_key(self._g_dim)]  # pyrefly: ignore[code]
      return rho * q_c * h
    else:
      return rho * q_c * self._dh

  def _air_molecules_per_area(
      self,
      p: ScalarField,
      vmr_h2o: ScalarField,
  ) -> ScalarField:
    """Computes the number of molecules in an atmospheric grid cell per area.

    The computation assumes the atmosphere to be in hydrostatic equilibrium.

    Args:
      p: The hydrostatic pressure variable.
      vmr_h2o: The volume mixing ratio of water vapor.

    Returns:
      A field containing the number of molecules of atmospheric gases per area
        [molecules/m**2].
    """
    dim_str = ('x', 'y', 'z')[self._g_dim]  # pyrefly: ignore[code]
    grad_kernel = ('kDx', 'kDy', 'kDz')[self._g_dim]  # pyrefly: ignore[code]
    dp = 0.5 * self._kernel_op.apply_kernel_op(p, grad_kernel, dim_str)
    mol_m_air = constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr_h2o
    return -(dp / constants.G) * constants.AVOGADRO / mol_m_air  # pyrefly: ignore[code]

  def _prepare_states(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> dict[str, Any]:
    """Prepares the states for the two-stream radiative transfer solver."""
    assert (
        'rho' in states
    ), 'RRTMGP requires the density (`rho`) to be present in `states`.'
    assert 'q_t' in states, (
        'RRTMGP requires the total specific humidity (`q_t`) to be present in'
        ' `states`.'
    )
    assert 'T' in additional_states, (
        'RRTMGP requires the temperature (`T`) to be present in'
        ' `additional_states`.'
    )
    # On very rare occasions, the total-water specific humidity may be a small
    # negative value, likely because the source term is not bounded properly.
    # This is usually innocuous for the evolution of the transport equation, but
    # here may lead to negative water vapor, which leads to negative relative
    # abundance of certain gas species. This results in negative interpolations
    # of quantities like optical depth and Planck fraction that are inherently
    # nonnegative. As a precaution, we clip the total-water specific humidity at
    # 0.
    q_t = jnp.maximum(states['q_t'], 0.0)

    # Condensed phase specific humidity required for cloud optics.
    if 'q_c' in additional_states:
      q_c = additional_states['q_c']
      liq_frac = self._water.liquid_fraction(additional_states['T'])
      q_liq = liq_frac * q_c
      q_ice = q_c - q_liq
    else:
      q_liq, q_ice = self._water.equilibrium_phase_partition(
          additional_states['T'], states['rho'], q_t
      )
      q_c = q_liq + q_ice

    pressure = self._water.p_ref(
        additional_states[self._vertical_coord_name], additional_states
    )

    # Reconstructs volume mixing ratio (vmr) fields of relevant gas species.
    vmr_lib = self._atmospheric_state.vmr
    vmr_fields = vmr_lib.reconstruct_vmr_fields_from_pressure(pressure)

    # Derive the water vapor vmr from the simulation state itself.
    vmr_fields.update(
        {'h2o': self._water.humidity_to_volume_mixing_ratio(q_t, q_c)}
    )
    molecules_per_area = self._air_molecules_per_area(
        pressure, vmr_fields['h2o']
    )
    lwp = self._compute_cloud_path(states['rho'], q_liq, additional_states)
    iwp = self._compute_cloud_path(states['rho'], q_ice, additional_states)
    cloud_r_eff_liq = self._microphysics_lib.cloud_particle_effective_radius(
        states['rho'], q_liq, 'l'
    )
    cloud_r_eff_ice = self._microphysics_lib.cloud_particle_effective_radius(
        states['rho'], q_ice, 'i'
    )
    return dict(
        pressure=pressure,
        temperature=additional_states['T'],
        molecules=molecules_per_area,
        vmr_fields=vmr_fields,
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_path_liq=lwp,
        cloud_r_eff_ice=cloud_r_eff_ice,
        cloud_path_ice=iwp,
    )

  def _clear_sky_states(
      self,
      states: dict[str, Any],
  ) -> dict[str, Any]:
    """Removes all the cloud states from the input states."""
    cloud_state_names = (
        'cloud_r_eff_liq',
        'cloud_path_liq',
        'cloud_r_eff_ice',
        'cloud_path_ice',
    )
    return {k: v for k, v in states.items() if k not in cloud_state_names}

  def compute_heating_rate(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      replicas: np.ndarray,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      sfc_temperature: ScalarField | float | None = None,
  ) -> dict[str, ScalarField]:
    """Computes the local heating rate due to radiative transfer.

    The optical properties of the layered atmosphere are computed using RRTMGP
    and the two-stream radiative transfer equation is solved for the net fluxes
    at the atmospheric grid cell faces. Based on the overall net radiative flux
    of the grid cell, a local heating rate is determined.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds all flow field variables and must include
        the total specific humidity (`q_t`) and density (`rho`).
      additional_states: A dictionary that holds all helper variables and must
        include temperature (`T`), and the vertical coordinates (`zz`) if the
        reference states is height dependent.
      sfc_temperature: The optional surface temperature [K] represented as
        either a 3D field having a single vertical dimension or as a scalar.

    Returns:
      A dictionary containing any subset of the following entries, as long as
      the corresponding keys are present in `additional_states`:
      'rad_heat_src' -> The heating rate due to radiative transfer, in K/s.
      'rad_flux_lw' -> The net longwave radiative flux, in W/m**2.
      'rad_flux_sw' -> The net shortwave radiative flux, W/m**2.
      'rad_flux_lw_clear' -> The net longwave radiative flux with cloud effects
        removed, in W/m**2.
      'rad_flux_sw_clear' -> The net shortwave radiative flux with cloud effects
        removed, in W/m**2.
    """
    primary_grid_states = self._prepare_states(states, additional_states)

    lw_fluxes = self._two_stream_solver.solve_lw(
        mesh,
        grid_params,
        replicas,
        **primary_grid_states,
        sfc_temperature=sfc_temperature,
    )
    sw_fluxes = self._two_stream_solver.solve_sw(
        mesh,
        grid_params,
        replicas,
        **primary_grid_states,
    )
    flux_net = lw_fluxes['flux_net'] + sw_fluxes['flux_net']
    # Heating rate in (K / s).
    heating_rate = self._two_stream_solver.compute_heating_rate(
        flux_net, primary_grid_states['pressure']
    )
    output: dict[str, ScalarField] = {
        rrtmgp_common.KEY_STORED_RADIATION: heating_rate
    }
    rrtmgp_keys = rrtmgp_common.additional_keys(
        self._config.radiative_transfer, self._config.additional_state_keys  # pyrefly: ignore[code]
    )
    # Select only the diagnostic flux keys.
    diagnostic_flux_keys = [
        k
        for k in rrtmgp_keys
        if k not in rrtmgp_common.required_keys(self._config.radiative_transfer)  # pyrefly: ignore[code]
    ]

    if not diagnostic_flux_keys:
      return output

    # Construct input states with cloud properties removed in case clear sky
    # fluxes are requested.
    primary_grid_states_clr = self._clear_sky_states(primary_grid_states)

    # Get net fluxes (upwelling - downwelling).
    for k in diagnostic_flux_keys:
      if k in (rrtmgp_common.KEY_RADIATIVE_FLUX_LW,):
        output[k] = lw_fluxes['flux_net']
      elif k in (rrtmgp_common.KEY_RADIATIVE_FLUX_SW,):
        output[k] = sw_fluxes['flux_net']
      # Compute clear sky fluxes by removing all cloud water and executing a
      # second pass of the two-stream solver.
      elif k in (rrtmgp_common.KEY_RADIATIVE_FLUX_LW_CLEAR,):
        lw_fluxes_clear = self._two_stream_solver.solve_lw(
            mesh,
            grid_params,
            replicas,
            **primary_grid_states_clr,
            sfc_temperature=sfc_temperature,
        )
        output[k] = lw_fluxes_clear['flux_net']
      elif k in (rrtmgp_common.KEY_RADIATIVE_FLUX_SW_CLEAR,):
        sw_fluxes_clear = self._two_stream_solver.solve_sw(
            mesh,
            grid_params,
            replicas,
            **primary_grid_states_clr,
        )
        output[k] = sw_fluxes_clear['flux_net']
      else:
        raise ValueError(f'Unknown RRTMGP flux key: {k}')

    return output
