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

R"""The radiative transfer equation solver (JAX).

JAX port of `swirl_lm.physics.radiation.rte.monochromatic_two_stream`.

Common symbols used in protected methods:
ssa: single-scattering albedo;
tau: optical depth;
g: asymmetry factor;
sw: shortwave;
lw: longwave;
gamma: exchange rate coefficient in the radiative transfer equation;
zenith: the zenith angle of collimated solar radiation.

References:
1. Shonk, Jonathan & Hogan, Robin. (2008). Tripleclouds: An Efficient Method for
   Representing Horizontal Cloud Inhomogeneity in 1D Radiation Schemes by Using
   Three Regions at Each Height. J. Climate. 21. 10.1175/2007JCLI1940.1.
2. Toon, Owen & McKay, C & Ackerman, T. & Santhanam, K.. (1989). Rapid
   calculation of radiative heating rates and photodissociation rates in
   Inhomogeneous multiple scattering atmospheres. Journal of Geophysical
   Research. 94. 10.1029/JD094iD13p16287.
3. Meador, W. E., and W. R. Weaver, 1980: Two-Stream Approximations to Radiative
   Transfer in Planetary Atmospheres: A Unified Description of Existing Methods
   and a New Improvement. J. Atmos. Sci., 37, 630-643.
4. Ukkonen, P. & Hogan, R. J. (2024) Twelve times faster yet accurate: A new
   state-of-the-art in radiation schemes via performance and spectral
   optimization. Journal of Advances in Modeling Earth Systems, 16(1). Portico.
"""


import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
import numpy as np
from swirl_lm.jax.physics.radiation.rte import rte_utils as utils
from swirl_lm.jax.utility import common_ops
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap
X0_KEY = utils.X0_KEY
PRIMARY_GRID_KEY = utils.PRIMARY_GRID_KEY
EXTENDED_GRID_KEY = utils.EXTENDED_GRID_KEY

# Secant of the longwave diffusivity angle per Fu et al. (1997).
_LW_DIFFUSIVE_FACTOR = 1.66
_EPSILON = 1e-6
# Minimum longwave optical depth required for nonzero source.
_MIN_TAU_FOR_LW_SRC = 1e-4
# Minimum value of the k parameter used in the transmittance.
_K_MIN = 1e-2


class MonochromaticTwoStreamSolver:
  """A library for solving the monochromatic two-stream radiative transfer."""

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      g_dim: int,
  ):
    self.halos = params.halo_width
    self.g_dim = g_dim
    self.rte_utils = utils.RTEUtils(params)
    self._kernel_op = kernel_op
    self._dim_str = ('x', 'y', 'z')[g_dim]
    self._shift_up_name = 'shift_up'
    self._shift_down_name = 'shift_dn'

  def _shift_up_fn(self, f: ScalarField) -> ScalarField:
    """Shifts the field up along the vertical dimension."""
    return self._kernel_op.apply_kernel_op(
        f, self._shift_up_name, self._dim_str
    )

  def _shift_down_fn(self, f: ScalarField) -> ScalarField:
    """Shifts the field down along the vertical dimension."""
    return self._kernel_op.apply_kernel_op(
        f, self._shift_down_name, self._dim_str
    )

  def lw_combine_sources(
      self,
      planck_srcs: ScalarFieldMap,
  ) -> dict[str, ScalarField]:
    """Combines the longwave source functions at each cell face.

    RRTMGP provides two source functions at each cell interface using the
    spectral mapping of each adjacent layer. These source functions are combined
    here via a geometric mean, and the result can be used for two-stream
    calculations.

    Args:
      planck_srcs: A dictionary containing the longwave Planck sources at the
        cell interfaces. The `level_planck_src_top` 3D variable contains the
        Planck source at the top cell face derived from the cell center's
        spectral mapping while the `level_planck_src_bottom` 3D variable
        contains the Planck source at the bottom cell face.

    Returns:
      A map of 3D variables for the combined Planck sources at the top face and
      the bottom cell face.
    """
    planck_src_top = planck_srcs['planck_src_top']
    planck_src_bottom = planck_srcs['planck_src_bottom']

    combined_src_top = jnp.sqrt(
        planck_src_top * self._shift_down_fn(planck_src_bottom)
    )
    combined_src_bottom = self._shift_up_fn(combined_src_top)

    return {
        'planck_src_top': combined_src_top,
        'planck_src_bottom': combined_src_bottom,
    }

  def _k_fn(self, gamma1: jax.Array, gamma2: jax.Array) -> jax.Array:
    """Computes the k parameter used in the transmittance."""
    k = jnp.sqrt(jnp.maximum((gamma1 + gamma2) * (gamma1 - gamma2), _EPSILON))
    return jnp.maximum(k, _K_MIN)

  def _rt_denominator_direct(
      self,
      gamma1: jax.Array,
      gamma2: jax.Array,
      tau: jax.Array,
      ssa: jax.Array,
      zenith: float,
  ) -> jax.Array:
    """Shared denominator of direct reflectance and transmittance functions."""
    k = self._k_fn(gamma1, gamma2)
    denom = self._rt_denominator_diffuse(gamma1, gamma2, tau)
    k_mu_squared = (k * jnp.cos(zenith)) ** 2

    # Equation 14, multiplying top and bottom by exp(-k*tau) and rearranging to
    # avoid division by 0.
    return jnp.where(
        jnp.abs(1.0 - k_mu_squared) >= _EPSILON,
        denom * (1.0 - k_mu_squared) / ssa,
        denom * _EPSILON / ssa,
    )

  def _direct_reflectance(
      self,
      gamma1: jax.Array,
      gamma2: jax.Array,
      gamma3: jax.Array,
      alpha2: jax.Array,
      tau: jax.Array,
      ssa: jax.Array,
      zenith: float,
  ) -> jax.Array:
    """Direct solar radiation reflectance (equation 14 of Meador and Weaver)."""
    k = self._k_fn(gamma1, gamma2)
    denom = self._rt_denominator_direct(gamma1, gamma2, tau, ssa, zenith)
    k_mu = k * jnp.cos(zenith)

    # Transmittance of direct, unscattered beam.
    t0 = jnp.exp(-tau / jnp.cos(zenith))

    # Equation 14 of Meador and Weaver (1980), multiplying top and bottom by
    # exp(-k*tau) and rearranging to avoid division by 0.
    exp_minusktau = jnp.exp(-k * tau)
    exp_minus2ktau = jnp.exp(-2.0 * k * tau)
    return (
        (1.0 - k_mu) * (alpha2 + k * gamma3)
        - (1.0 + k_mu) * (alpha2 - k * gamma3) * exp_minus2ktau
        - 2.0 * (k * gamma3 - alpha2 * k_mu) * exp_minusktau * t0
    ) / denom

  def _direct_transmittance(
      self,
      gamma1: jax.Array,
      gamma2: jax.Array,
      gamma4: jax.Array,
      alpha1: jax.Array,
      tau: jax.Array,
      ssa: jax.Array,
      zenith: float,
  ) -> jax.Array:
    """Direct solar radiation transmittance (Meador and Weaver, equation 15)."""
    k = self._k_fn(gamma1, gamma2)
    denom = self._rt_denominator_direct(gamma1, gamma2, tau, ssa, zenith)
    k_mu = k * jnp.cos(zenith)
    k_y4 = k * gamma4

    # Transmittance of direct, unscattered beam.
    t0 = jnp.exp(-tau / jnp.cos(zenith))

    exp_minusktau = jnp.exp(-k * tau)
    exp_minus2ktau = jnp.exp(-2.0 * k * tau)

    # Equation 15 (Meador and Weaver (1980)), refactored for numerical stability
    # by 1) multiplying top and bottom by exp(-k*tau), 2) multiplying through by
    # exp(-tau/mu0) to prefer underflow to overflow, and 3) omitting direct
    # transmittance.
    return (
        -(
            (1.0 + k_mu) * (alpha1 + k_y4) * t0
            - (1.0 - k_mu) * (alpha1 - k_y4) * exp_minus2ktau * t0
            - 2.0 * (k_y4 + alpha1 * k_mu) * exp_minusktau
        )
        / denom
    )

  def _rt_denominator_diffuse(
      self, gamma1: jax.Array, gamma2: jax.Array, tau: jax.Array
  ) -> jax.Array:
    """The shared denominator of the diffuse reflectance and transmittance."""
    # As in the original RRTMGP Fortran code, this expression has been
    # refactored to avoid rounding errors when k, gamma1 are of very different
    # magnitudes.
    k = self._k_fn(gamma1, gamma2)
    return k * (1 + jnp.exp(-2.0 * tau * k)) + gamma1 * (
        1 - jnp.exp(-2.0 * tau * k)
    )

  def _diffuse_reflectance(
      self, gamma1: jax.Array, gamma2: jax.Array, tau: jax.Array
  ) -> jax.Array:
    """The diffuse reflectance (equation 25 of Meador and Weaver (1980))."""
    k = self._k_fn(gamma1, gamma2)
    denom = self._rt_denominator_diffuse(gamma1, gamma2, tau)
    return gamma2 * (1.0 - jnp.exp(-2.0 * tau * k)) / denom

  def _diffuse_transmittance(
      self, gamma1: jax.Array, gamma2: jax.Array, tau: jax.Array
  ) -> jax.Array:
    """The diffuse transmittance (equation 26 of Meador and Weaver (1980))."""
    k = self._k_fn(gamma1, gamma2)
    denom = self._rt_denominator_diffuse(gamma1, gamma2, tau)
    return 2.0 * k * jnp.exp(-tau * k) / denom

  def lw_cell_source_and_properties(
      self,
      optical_depth: jax.Array,
      ssa: jax.Array,
      level_src_bottom: jax.Array,
      level_src_top: jax.Array,
      asymmetry_factor: jax.Array,
  ) -> dict[str, ScalarField]:
    """Computes longwave two-stream reflectance, transmittance, and sources.

    The upwelling and downwelling Planck functions and the optical properties
    (transmission and reflectance) are calculated at the cell centers. Equations
    are developed in Meador and Weaver (1980) and Toon et al. (1989).

    Args:
      optical_depth: The pointwise optical depth.
      ssa: The pointwise single-scattering albedo.
      level_src_bottom: The Planck source at the bottom cell face [W / m**2 /
        sr].
      level_src_top: The Planck source at the top cell face [W / m**2 / sr].
      asymmetry_factor: The pointwise asymmetry factor.

    Returns:
      A dictionary containing the following items:
      'reflectance': A 3D variable containing the pointwise reflectance.
      'transmittance': A 3D variable containing the pointwise transmittance.
      'src_up': A 3D variable containing the pointwise upwelling Planck source.
      'src_down': A 3D variable with the pointwise downwelling Planck source.
    """
    gamma1 = _LW_DIFFUSIVE_FACTOR * (1 - 0.5 * ssa * (1.0 + asymmetry_factor))
    gamma2 = _LW_DIFFUSIVE_FACTOR * 0.5 * ssa * (1.0 - asymmetry_factor)

    r_diff = self._diffuse_reflectance(gamma1, gamma2, optical_depth)
    t_diff = self._diffuse_transmittance(gamma1, gamma2, optical_depth)

    # From Toon et al. (JGR 1989) Eqs 26-27, first-order coefficient of the
    # Taylor series expansion of the Planck function in terms of the optical
    # depth.
    b_1 = (level_src_bottom - level_src_top) / (
        optical_depth * (gamma1 + gamma2)
    )

    # Compute longwave source function for upward and downward emission at cell
    # interfaces using linear-in-tau assumption.
    c_up_top = b_1 + level_src_top
    c_up_bottom = b_1 + level_src_bottom
    c_down_top = -b_1 + level_src_top
    c_down_bottom = -b_1 + level_src_bottom

    def cell_center_src_fn(
        downstream_out: jax.Array,
        downstream_in: jax.Array,
        upstream_in: jax.Array,
        refl: jax.Array,
        tran: jax.Array,
        tau: jax.Array,
    ) -> jax.Array:
      """Computes the flux at the cell center consistent with face fluxes.

      The cell center source is the residual that remains when one subtracts
      from the downstream outward flux two contributions:
      1. the upstream inward flux that is transmitted through the cell and
      2. the downstream inward flux that is reflected off the cell.

      Args:
        downstream_out: Downstream outward flux.
        downstream_in: Downstream inward flux.
        upstream_in: Upstream inward flux.
        refl: The grid cell reflectance.
        tran: The grid cell transmittance.
        tau: The grid cell optical depth.

      Returns:
        The directional radiative source at the cell center consistent with the
        given face sources [W / m**2].
      """
      src = math.pi * (
          downstream_out - refl * downstream_in - tran * upstream_in
      )
      # Filter out sources where the optical depth is too small.
      return jnp.where(tau > _MIN_TAU_FOR_LW_SRC, src, jnp.zeros_like(src))

    src_up = cell_center_src_fn(
        c_up_top,
        c_down_top,
        c_up_bottom,
        r_diff,
        t_diff,
        optical_depth,
    )
    src_down = cell_center_src_fn(
        c_down_bottom,
        c_up_bottom,
        c_down_top,
        r_diff,
        t_diff,
        optical_depth,
    )
    return {
        't_diff': t_diff,
        'r_diff': r_diff,
        'src_up': src_up,
        'src_down': src_down,
    }

  def sw_cell_properties(
      self,
      zenith: float,
      optical_depth: ScalarField,
      ssa: ScalarField,
      asymmetry_factor: ScalarField,
  ) -> dict[str, ScalarField]:
    """Computes shortwave reflectance and transmittance.

    Two-stream solutions to direct and diffuse reflectance and transmittance as
    a function of optical depth, single-scattering albedo, and asymmetry factor.
    Equations are developed in Meador and Weaver (1980).

    Args:
      zenith: The zenith angle of the shortwave collimated radiation.
      optical_depth: A 3D variable containing the pointwise optical depth.
      ssa: A 3D variable containing the pointwise single-scattering albedo.
      asymmetry_factor: A 3D variable containing the pointwise asymmetry factor.

    Returns:
      A dictionary containing the following items:
      't_diff': A 3D variable containing the diffuse transmittance.
      'r_diff': A 3D variable containing the diffuse reflectance.
      't_dir': A 3D variable containing the direct transmittance.
      'r_dir': A 3D variable containing the direct reflectance.
    """
    # Exchange rate coefficients from Zdunkowski et al. (1980).
    gamma1 = (8.0 - ssa * (5.0 + 3.0 * asymmetry_factor)) * 0.25
    gamma2 = 3.0 * ssa * (1.0 - asymmetry_factor) * 0.25
    gamma3 = (2.0 - 3.0 * jnp.cos(zenith) * asymmetry_factor) * 0.25
    gamma4 = 1.0 - gamma3
    alpha1 = gamma1 * gamma4 + gamma2 * gamma3
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4

    # Diffuse reflectance and transmittance.
    r_diff = self._diffuse_reflectance(gamma1, gamma2, optical_depth)
    t_diff = self._diffuse_transmittance(gamma1, gamma2, optical_depth)

    # Direct reflectance and transmittance.
    r_dir_unconstrained = self._direct_reflectance(
        gamma1,
        gamma2,
        gamma3,
        alpha2,
        optical_depth,
        ssa,
        zenith,
    )
    t_dir_unconstrained = self._direct_transmittance(
        gamma1,
        gamma2,
        gamma4,
        alpha1,
        optical_depth,
        ssa,
        zenith,
    )

    # Constrain reflectance and transmittance to be positive and to not go above
    # physical limits by enforcing the constraint that the direct beam can
    # either be reflected, penetrate unscattered to the bottom of the grid
    # cell, or penetrate through but be scattered on the way.

    # Direct transmittance of unscattered beam.
    t0 = jnp.exp(-optical_depth / jnp.cos(zenith))

    # Equation 9 of Hogan and Ukonnen (2024).
    r_dir = jnp.clip(r_dir_unconstrained, 0.0, 1.0 - t0)
    # Equation 10 of Hogan and Ukonnen (2024).
    t_dir = jnp.clip(t_dir_unconstrained, 0.0, 1.0 - t0 - r_dir)

    return {
        't_diff': t_diff,
        'r_diff': r_diff,
        't_dir': t_dir,
        'r_dir': r_dir,
    }

  def sw_cell_source(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      replicas: np.ndarray,
      t_dir: ScalarField,
      r_dir: ScalarField,
      optical_depth: ScalarField,
      toa_flux: ScalarField,
      sfc_albedo_direct: ScalarField,
      zenith: float,
  ) -> dict[str, dict[str, ScalarField]]:
    """Computes monochromatic shortwave direct-beam flux and diffuse source.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      replicas: The mapping from the core coordinate to the local replica id.
      t_dir: A 3D variable for the direct-beam transmittance.
      r_dir: A 3D variable for the direct-beam reflectance.
      optical_depth: A 3D variable for the optical depth.
      toa_flux: The top of atmosphere incoming flux represented by a 2D plane.
      sfc_albedo_direct: The surface albedo with respect to direct radiation.
      zenith: The zenith solar angle.

    Returns:
      A dictionary containing a dict for the primary grid with key 'primary'.
      Each dict contains the following items:
      'src_up': A 3D variable for the cell center upward source.
      'src_down': A 3D variable for the cell center downward source.
      'flux_down_dir': A 3D variable for the solved downwelling direct-beam
        radiative flux at the bottom cell face.
      'sfc_src': A 2D variable for the shortwave source emanating from surface.
    """
    t_noscat = jnp.exp(-optical_depth / jnp.cos(zenith))
    mu = jnp.cos(zenith)

    # The vertical component of incident flux at the top boundary.
    flux_down_direct_bc = toa_flux * mu

    # Global recurrent accumulation for the direct-beam downward flux at the
    # bottom cell face unraveling from the top of the atmosphere down to the
    # surface. The recurrence follows the simple relation:
    # flux_down_direct[i] = T_no_scatter[i] * flux_down_direct[i + 1]
    op = lambda w, x0: w * x0
    kwargs: dict[str, ScalarField] = {
        'w': t_noscat,
        X0_KEY: flux_down_direct_bc,
    }

    flux_down_direct = self.rte_utils.cumulative_recurrent_op(
        mesh,
        grid_params,
        replicas,
        op,
        kwargs,
        dim=self.g_dim,
        forward=False,
    )

    # Upward source from direct-beam reflection at the cell center.
    src_up = r_dir * self._shift_down_fn(flux_down_direct[PRIMARY_GRID_KEY])

    # Downward source from direct-beam transmittance at the cell center.
    src_down = t_dir * self._shift_down_fn(flux_down_direct[PRIMARY_GRID_KEY])

    # Direct-beam flux incident on the surface.
    flux_down_sfc = common_ops.slice_field(
        flux_down_direct[PRIMARY_GRID_KEY], self.g_dim, self.halos, size=1
    )
    core_idx = utils._get_core_coordinate(replicas, self.g_dim)  # pylint: disable=protected-access

    # The surface source is the direct-beam downard flux that is reflected from
    # the surface.
    sfc_src = jax.lax.cond(
        core_idx == 0,
        lambda: flux_down_sfc * sfc_albedo_direct,
        lambda: jnp.zeros_like(flux_down_sfc),
    )

    srcs_primary: dict[str, ScalarField] = {
        'src_up': src_up,
        'src_down': src_down,
        'flux_down_dir': flux_down_direct[PRIMARY_GRID_KEY],
        'sfc_src': sfc_src,
    }

    return {PRIMARY_GRID_KEY: srcs_primary}

  def _solve_rte_2stream(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      replicas: np.ndarray,
      t_diff: ScalarField,
      r_diff: ScalarField,
      src_up: ScalarField,
      src_down: ScalarField,
      top_flux_down: ScalarField,
      sfc_emission: ScalarField,
      sfc_reflectance: ScalarField,
  ) -> dict[str, ScalarField]:
    r"""Solves the monochromatic two-stream radiative transfer equation.

    Given boundary conditions for the downward flux at the top of the atmosphere
    (`top_flux_down`) and the upward surface emission (`sfc_emission`), this
    computes the two-stream approximation of the upwelling and downwelling
    radiative fluxes at the cell faces based on the equations of Shonk and Hogan
    (2008), doi:10.1175/2007JCLI1940.1. All the computations here assume a
    single absorption interval (or `g` interval in RRTM nomenclature). This
    function needs to be applied to each `g` interval separately.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      replicas: The mapping from the core coordinate to the local replica id.
      t_diff: A 3D variable containing the cell center transmittance.
      r_diff: A 3D variable containing the cell center reflectance.
      src_up: A 3D variable containing the cell center upward emission.
      src_down: A 3D variable containing the cell center downward emission.
      top_flux_down: The downward component of the incoming flux at the top
        boundary of the atmosphere.
      sfc_emission: The upward surface emission.
      sfc_reflectance: The surface reflectance.

    Returns:
      A dictionary containing fluxes at the bottom cell face:
      'flux_up' -> The upwelling radiative flux.
      'flux_down' -> The downwelling radiative flux.
    """

    def global_recurrent_op(
        kwargs: dict[str, ScalarField],
        forward: bool,
        op: Callable[..., ScalarField],
    ) -> dict[str, ScalarField]:
      return self.rte_utils.cumulative_recurrent_op(
          mesh,
          grid_params,
          replicas,
          op,
          kwargs,
          dim=self.g_dim,
          forward=forward,
      )

    # Global recurrent accumulation for the albedo of the atmosphere below a
    # certain level, computed from the surface all the way to the top boundary.
    # The recurrence relation for albedo is taken from Shonk and Hogan Equation
    # 9.

    def albedo_op(
        r_diff: jax.Array,
        t_diff: jax.Array,
        x0: jax.Array,
    ) -> jax.Array:
      """Recurrent formula for albedo solution unraveling from the surface."""
      albedo_below = x0
      # Geometric series solution accounting for infinite reflection events.
      beta = 1.0 / (1.0 - r_diff * albedo_below)
      return r_diff + t_diff**2 * beta * albedo_below

    albedo_vars: dict[str, ScalarField] = {
        'r_diff': r_diff,
        't_diff': t_diff,
        X0_KEY: sfc_reflectance,
    }
    albedo = global_recurrent_op(
        albedo_vars,
        forward=True,
        op=albedo_op,
    )

    # Global recurrent accumulation for the aggregate upwelling source emission
    # computed from the surface all the way to the top of the atmosphere.
    # The coefficient and bias terms of the recurrence relation for emission are
    # taken from Shonk and Hogan Equation 11.

    def upward_emission_op(
        src_up: jax.Array,
        src_down: jax.Array,
        t_diff: jax.Array,
        r_diff: jax.Array,
        albedo: jax.Array,
        x0: jax.Array,
    ) -> jax.Array:
      """Recurrent formula for upward emission starting from the surface."""
      emission_from_below = x0
      # Geometric series solution accounting for infinite reflection events.
      beta = 1.0 / (1.0 - r_diff * albedo)
      return src_up + t_diff * beta * (emission_from_below + src_down * albedo)

    emission_vars: dict[str, ScalarField] = {
        'src_up': src_up,
        'src_down': src_down,
        't_diff': t_diff,
        'r_diff': r_diff,
        'albedo': self._shift_up_fn(albedo[PRIMARY_GRID_KEY]),
        X0_KEY: sfc_emission,
    }
    emiss_up = global_recurrent_op(
        emission_vars,
        forward=True,
        op=upward_emission_op,
    )

    # Global recurrent accumulation for the downwelling radiative flux solution
    # at the bottom face, unravelling from the top of the atmosphere down to the
    # surface. The coefficient and bias terms are taken from Shonk and Hogan
    # Equation 13.

    def flux_down_op(
        emiss_up: jax.Array,
        src_down: jax.Array,
        t_diff: jax.Array,
        r_diff: jax.Array,
        albedo: jax.Array,
        x0: jax.Array,
    ) -> jax.Array:
      """Recurrent formula for downwelling flux initiating at top boundary."""
      flux_dn_from_above = x0
      # Geometric series solution accounting for infinite reflection events.
      beta = 1.0 / (1.0 - r_diff * albedo)
      return (t_diff * flux_dn_from_above + r_diff * emiss_up + src_down) * beta

    flux_down_vars: dict[str, ScalarField] = {
        'emiss_up': self._shift_up_fn(emiss_up[PRIMARY_GRID_KEY]),
        'src_down': src_down,
        't_diff': t_diff,
        'r_diff': r_diff,
        'albedo': self._shift_up_fn(albedo[PRIMARY_GRID_KEY]),
        X0_KEY: top_flux_down,
    }
    flux_down = global_recurrent_op(
        flux_down_vars,
        forward=False,
        op=flux_down_op,
    )

    # The upwelling radiative flux at the bottom face.
    flux_up = flux_down[PRIMARY_GRID_KEY] * self._shift_up_fn(
        albedo[PRIMARY_GRID_KEY]
    ) + self._shift_up_fn(emiss_up[PRIMARY_GRID_KEY])

    return {
        'flux_up': flux_up,
        'flux_down': flux_down[PRIMARY_GRID_KEY],
    }

  def lw_transport(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      replicas: np.ndarray,
      t_diff: ScalarField,
      r_diff: ScalarField,
      src_up: ScalarField,
      src_down: ScalarField,
      top_flux_down: ScalarField,
      sfc_src: ScalarField,
      sfc_emissivity: ScalarField,
      **kwargs: Any,
  ) -> dict[str, ScalarField]:
    """Computes the monochromatic longwave diffusive flux of the atmosphere.

    The upwelling and downwelling fluxes are computed from the equations of
    Shonk and Hogan (2008, doi:10.1175/2007JCLI1940.1) assuming a single
    reflection event. The net flux is also computed at every face.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      replicas: The mapping from the core coordinate to the local replica id.
      t_diff: A 3D variable containing the cell center transmittance.
      r_diff: A 3D variable containing the cell center reflectance.
      src_up: A 3D variable containing the cell center Planck upward emission.
      src_down: A 3D variable containing the cell center Planck downward
        emission.
      top_flux_down: The downward flux at the top boundary of the atmosphere.
      sfc_src: The surface Planck source.
      sfc_emissivity: The surface emissivity.
      **kwargs: Additional keyword arguments (unused, for API compatibility).

    Returns:
      A dictionary containing fluxes at the bottom cell face [W/m**2]:
      'flux_up' -> The upwelling radiative flux.
      'flux_down' -> The downwelling radiative flux.
      'flux_net' -> The net radiative flux.
    """
    del kwargs  # Unused.

    # The source of diffuse radiation is the surface emission.
    sfc_emission = np.pi * sfc_emissivity * sfc_src
    # The surface reflectance is just the complement of the surface emissivity.
    sfc_reflectance = 1.0 - sfc_emissivity
    fluxes: dict[str, ScalarField] = {}
    fluxes.update(
        self._solve_rte_2stream(
            mesh,
            grid_params,
            replicas,
            t_diff,
            r_diff,
            src_up,
            src_down,
            top_flux_down,
            sfc_emission,
            sfc_reflectance,
        )
    )
    fluxes['flux_net'] = fluxes['flux_up'] - fluxes['flux_down']
    return fluxes

  def sw_transport(
      self,
      mesh: Mesh,
      grid_params: grid_parametrization.GridParametrization,
      replicas: np.ndarray,
      t_diff: ScalarField,
      r_diff: ScalarField,
      src_up: ScalarField,
      src_down: ScalarField,
      sfc_src: ScalarField,
      sfc_albedo: ScalarField,
      flux_down_dir: ScalarField,
      **kwargs: Any,
  ) -> dict[str, ScalarField]:
    """Computes the monochromatic shortwave fluxes in a layered atmosphere.

    The direct-beam downward flux `flux_down_dir` is added to the downwelling
    diffuse flux in the final solution.

    Args:
      mesh: The JAX device mesh.
      grid_params: Grid parametrization.
      replicas: The mapping from the core coordinate to the local replica id.
      t_diff: A 3D variable for the cell transmittance.
      r_diff: A 3D variable for the cell reflectance.
      src_up: A 3D variable for the cell center upward source.
      src_down: A 3D variable for the cell center downward source.
      sfc_src: A 2D variable for the direct-beam shortwave radiation reflected
        upward from the surface.
      sfc_albedo: The surface albedo.
      flux_down_dir: A 3D variable for the solved downwelling direct-beam
        radiative flux at the bottom cell face.
      **kwargs: Additional keyword arguments (unused, for API compatibility).

    Returns:
      A dictionary containing fluxes at the bottom cell face:
      'flux_up' -> The upwelling radiative flux.
      'flux_down' -> The downwelling radiative flux.
      'flux_net' -> The net radiative flux.
    """
    del kwargs  # Unused.

    fluxes: dict[str, ScalarField] = {}
    fluxes.update(
        self._solve_rte_2stream(
            mesh,
            grid_params,
            replicas,
            t_diff,
            r_diff,
            src_up,
            src_down,
            jnp.zeros_like(sfc_src),
            sfc_src,
            sfc_albedo,
        )
    )

    # Add the direct-beam contribution to the downwelling flux.
    fluxes['flux_down'] = fluxes['flux_down'] + flux_down_dir
    # The net flux computed at cell faces.
    fluxes['flux_net'] = fluxes['flux_up'] - fluxes['flux_down']

    return fluxes
