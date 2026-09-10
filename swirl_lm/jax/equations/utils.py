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
"""Utility functions commonly used in different equations.

This is the JAX port of `swirl_lm.equations.utils`. All operations work on
3D `jax.Array` directly (no list-of-2D-slices support).
"""


import functools
from typing import Callable

import jax
import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import monin_obukhov_similarity_theory as most_lib
from swirl_lm.jax.numerics import calculus
from swirl_lm.jax.numerics import derivatives
from swirl_lm.jax.numerics import filters
from swirl_lm.jax.numerics import interpolation
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import stretched_grid_util
from swirl_lm.jax.utility import types
from swirl_lm.physics import constants
from swirl_lm.physics.thermodynamics import thermodynamics_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# Parameters required by source terms due to subsidence velocity.
_W_MAX = -0.65e-2
_Z_F1 = 1500.0
_Z_F5 = 2100.0
_D = 3.75e-6

# Small number to avoid round of error issues when handling gravity.
_G_EPS = 1e-6


def shear_stress(
    deriv_lib: derivatives.Derivatives,
    mu: ScalarField,
    u: ScalarField,
    v: ScalarField,
    w: ScalarField,
    additional_states: ScalarFieldMap,
    shear_bc_update_fn: (
        dict[str, Callable[[ScalarField], ScalarField]] | None
    ) = None,
) -> ScalarFieldMap:
  """Computes the viscous shear stress at cell centers.

  The shear stress is computed as:
    tau_ij = mu * [du_i/dx_j + du_j/dx_i], i != j
    tau_ii = 2 * mu * [du_i/dx_i - 1/3 * du_k/dx_k * delta_ij]

  Args:
    deriv_lib: An instance of the derivatives library.
    mu: Dynamic viscosity of the flow field.
    u: Velocity component in the x dimension (with updated BC).
    v: Velocity component in the y dimension (with updated BC).
    w: Velocity component in the z dimension (with updated BC).
    additional_states: A dictionary containing helper variables.
    shear_bc_update_fn: An optional dictionary of halo_exchange functions for
      the shear stress tensor.

  Returns:
    The 9-component stress tensor for each grid point, keyed by 'xx', 'xy',
    etc. Values in the halo with width 1 are invalid.
  """
  du_dx = calculus.grad(deriv_lib, [u, v, w], additional_states)

  du_00 = du_dx[0][0]
  du_01 = du_dx[0][1]
  du_02 = du_dx[0][2]
  du_10 = du_dx[1][0]
  du_11 = du_dx[1][1]
  du_12 = du_dx[1][2]
  du_20 = du_dx[2][0]
  du_21 = du_dx[2][1]
  du_22 = du_dx[2][2]

  s00 = du_00
  s01 = 0.5 * (du_01 + du_10)
  s02 = 0.5 * (du_02 + du_20)
  s10 = s01
  s11 = du_11
  s12 = 0.5 * (du_12 + du_21)
  s20 = s02
  s21 = s12
  s22 = du_22

  div_u = du_00 + du_11 + du_22

  tau: dict[str, ScalarField] = {
      'xx': 2.0 * mu * (s00 - div_u / 3.0),
      'xy': 2.0 * mu * s01,
      'xz': 2.0 * mu * s02,
      'yx': 2.0 * mu * s10,
      'yy': 2.0 * mu * (s11 - div_u / 3.0),
      'yz': 2.0 * mu * s12,
      'zx': 2.0 * mu * s20,
      'zy': 2.0 * mu * s21,
      'zz': 2.0 * mu * (s22 - div_u / 3.0),
  }

  if shear_bc_update_fn:
    for key, fn in shear_bc_update_fn.items():
      tau[key] = fn(tau[key])

  return tau


def shear_flux(
    params: parameters_lib.SwirlLMParameters,
) -> Callable[..., ScalarFieldMap]:
  """Generates a function that computes shear fluxes at cell faces.

  If the Monin-Obukhov Similarity Theory (MOST) boundary model is configured,
  the surface shear stresses at the ground are replaced with MOST closures.

  Args:
    params: The simulation parameter context.

  Returns:
    A function that computes the 9-component shear stress tensor on faces.
  """
  if params.boundary_models is not None and params.boundary_models.HasField(
      'most'
  ):
    most = most_lib.monin_obukhov_similarity_theory_factory(params)
  else:
    most = None

  def shear_flux_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      deriv_lib: derivatives.Derivatives,
      mu: ScalarField,
      u: ScalarField,
      v: ScalarField,
      w: ScalarField,
      rho: ScalarField,
      helper_variables: ScalarFieldMap,
  ) -> ScalarFieldMap:
    """Computes the viscous shear stress on cell faces.

    The shear stress is computed as:
      tau_ij = mu * [du_i/dx_j + du_j/dx_i], i != j
      tau_ii = 2 * mu * [du_i/dx_i - 1/3 * du_k/dx_k * delta_ij]

    Locations of the fluxes:
      tau00/tau_xx: x face, i - 1/2 stored at i;
      tau01/tau_xy: y face, j - 1/2 stored at j;
      ...etc.

    Args:
      kernel_op: An object holding a library of kernel operations.
      deriv_lib: An instance of the derivatives library.
      mu: Dynamic viscosity of the flow field.
      u: Velocity component in the x dimension (with updated BC).
      v: Velocity component in the y dimension (with updated BC).
      w: Velocity component in the z dimension (with updated BC).
      rho: Density of the flow field.
      helper_variables: A dictionary containing helper variables.

    Returns:
      The 9-component stress tensor for each grid point on faces.
    """
    axes = params.grid_params.data_axis_order
    interp = functools.partial(
        interpolation.centered_node_to_face, kernel_op=kernel_op
    )

    def grad_interp(
        f: ScalarField,
        deriv_axis: str,
        interp_axis: str,
    ) -> ScalarField:
      """Computes derivative of `f` in `deriv_axis` on faces in `interp_axis`."""
      deriv_f = deriv_lib.deriv_centered(f, deriv_axis, helper_variables)
      return interp(deriv_f, interp_axis)

    velocity = {'x': u, 'y': v, 'z': w}

    # Compute s_ij on faces in dim j.
    s = {}
    for i_ax in axes:
      for j_ax in axes:
        if i_ax == j_ax:
          s[i_ax + j_ax] = deriv_lib.deriv_node_to_face(
              velocity[i_ax], j_ax, helper_variables
          )
        else:
          s[i_ax + j_ax] = 0.5 * (
              deriv_lib.deriv_node_to_face(
                  velocity[i_ax], j_ax, helper_variables
              )
              + grad_interp(velocity[j_ax], i_ax, j_ax)
          )

    # Divergence on each face direction.
    div_face = {}
    for face_ax in axes:
      div_face[face_ax] = s[face_ax + face_ax]
      for other_ax in axes:
        if other_ax != face_ax:
          div_face[face_ax] = div_face[face_ax] + grad_interp(
              velocity[other_ax], other_ax, face_ax
          )

    # Compute tau_ij on faces in dim j.
    tau: dict[str, ScalarField] = {}
    for i_ax in axes:
      for j_ax in axes:
        mu_face = interp(mu, j_ax)
        key = i_ax + j_ax
        if i_ax == j_ax:
          tau[key] = 2.0 * mu_face * (s[key] - div_face[j_ax] / 3.0)
        else:
          tau[key] = 2.0 * mu_face * s[key]

    # Replace ground-level shear stress with MOST closure if configured.
    if most is not None:
      theta = helper_variables.get('theta')
      if theta is None:
        raise ValueError('`theta` is missing for the MOST model.')

      helper_vars: dict[str, ScalarField] = {
          'u': u,
          'v': v,
          'w': w,
          'theta': theta,
          'rho': rho,
      }

      # Get the surface shear stress from MOST.
      tau_s1, tau_s2, _ = most.surface_shear_stress_and_heat_flux_update_fn(
          helper_vars
      )

      # The sign is reversed to be consistent with the diffusion scheme.
      tau_s1 = -tau_s1
      tau_s2 = -tau_s2

      # Determine which tau components to replace based on vertical_dim.
      physical_axes = ('x', 'y', 'z')
      g_dim = most.vertical_dim
      g_axis = physical_axes[g_dim]
      horiz_dims = most.horizontal_dims

      # Map horizontal dimensions to axis names.
      horiz_axes = [physical_axes[d] for d in horiz_dims]

      # tau_s1 corresponds to shear stress of first horizontal velocity
      # on g_axis face, tau_s2 for the second horizontal velocity.
      tau_key_1 = horiz_axes[0] + g_axis
      tau_key_2 = horiz_axes[1] + g_axis

      # Replace the ground-level plane in the tau arrays.
      # The ground is at face=0, index=halo_width.
      halo_width = params.halo_width
      axis_index: int = params.grid_params.get_axis_index(g_axis)  # pyrefly: ignore[bad-assignment]

      # `jax.lax.axis_index` raises `NameError` outside of a parallel context
      # (e.g. un-sharded CPU unit tests). `is_bottom_shard = True` is required
      # in un-sharded execution because a single device holds the entire domain
      # including the ground boundary.
      try:
        is_bottom_shard = jax.lax.axis_index(g_axis) == 0
      except NameError:
        is_bottom_shard = True

      for tau_key, tau_s in ((tau_key_1, tau_s1), (tau_key_2, tau_s2)):
        # Expand the 2D surface value back to 3D (single plane).
        plane = jnp.expand_dims(tau_s, axis=axis_index)
        # Build the start indices for dynamic_update_slice.
        start_idx = [0, 0, 0]
        start_idx[axis_index] = halo_width
        tau_updated = jax.lax.dynamic_update_slice(
            tau[tau_key], plane, tuple(start_idx)
        )
        tau[tau_key] = jnp.where(is_bottom_shard, tau_updated, tau[tau_key])

    return tau

  return shear_flux_fn


def bound_viscosity(
    nu: float | ScalarField,
    additional_states: ScalarFieldMap,
    params: parameters_lib.SwirlLMParameters,
) -> float | ScalarField:
  """Sets an upper bound to `nu` following the stability constraint."""
  if params.diff_stab_crit is None:
    return nu

  axes = params.grid_params.data_axis_order
  for dim, axis in enumerate(axes):
    if params.use_stretched_grid[dim]:
      physical_dim = ('x', 'y', 'z').index(axis)
      h = additional_states[stretched_grid_util.h_face_key(physical_dim)]
    else:
      h = params.grid_spacings[dim]
    nu_max = params.diff_stab_crit * (h**2 / params.dt)
    nu = jnp.minimum(nu, nu_max)

  return nu


def subsidence_velocity_stevens(zz: ScalarField) -> ScalarField:
  """Computes the subsidence velocity following the Stevens formulation."""
  return -_D * zz


def subsidence_velocity_siebesma(zz: ScalarField) -> ScalarField:
  """Computes the subsidence velocity following the Siebesma formulation."""
  w = jnp.where(
      zz <= _Z_F1,
      _W_MAX * zz / _Z_F1,
      _W_MAX * (1.0 - (zz - _Z_F1) / (_Z_F5 - _Z_F1)),
  )
  return jnp.where(zz <= _Z_F5, w, jnp.zeros_like(w))


def source_by_subsidence_velocity(
    deriv_lib: derivatives.Derivatives,
    rho: ScalarField,
    height: ScalarField,
    field: ScalarField,
    vertical_axis: str,
    additional_states: ScalarFieldMap,
) -> ScalarField:
  """Computes the source term for `field` due to subsidence velocity.

  Args:
    deriv_lib: An instance of the derivatives library.
    rho: The density of the flow field.
    height: The coordinates in the direction vertical to the ground.
    field: The quantity to which the source term is computed.
    vertical_axis: The vertical axis aligned with gravity (e.g., 'z').
    additional_states: A dictionary that holds all helper variables.

  Returns:
    The source term for `field` due to the subsidence velocity.
  """
  df_dh = deriv_lib.deriv_centered(field, vertical_axis, additional_states)
  w = subsidence_velocity_stevens(height)
  return -rho * w * df_dh


def buoyancy_source(
    rho: ScalarField,
    rho_0: ScalarField,
    params: parameters_lib.SwirlLMParameters,
    dim: int,
    additional_states: ScalarFieldMap,
) -> ScalarField:
  """Computes the gravitational force of the momentum equation.

  Args:
    rho: The density of the flow field.
    rho_0: The reference density of the environment.
    params: The simulation parameter context. `thermodynamics.solver_mode` is
      used here.
    dim: The spatial dimension that this source corresponds to.
    additional_states: Mapping that contains the optional scale factors.

  Returns:
    The source term of the momentum equation due to buoyancy.
  """
  g_dim = params.gravity_direction[dim]
  if abs(g_dim) < _G_EPS:
    return jnp.zeros_like(rho)

  if params.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
    drho = (rho - rho_0) * rho_0 / rho
  else:
    drho = rho - rho_0

  # Filter the density difference.
  drho = filters.filter_op(
      params.kernel_op, params.grid_params, drho, additional_states, order=2
  )

  return drho * g_dim * constants.G
