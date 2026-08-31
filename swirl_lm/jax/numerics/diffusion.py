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
"""A library that computes the diffusion term in the Navier-Stokes solver.

The diffusion term for scalars has 3 components only, in each component the 2
first order derivatives are performed along the same direction, i.e. d/dx(d/dx),
d/dy(d/dy), and d/dz(d/dz).

The diffusion term for velocity considers not only derivatives along same
directions, but also in perpendicular directions, e.g. d/dy(d/dx). There are 3
methods to compute these terms:

DIFFUSION_SCHEME_CENTRAL_5: both the inner and outer first order derivatives are
computed with 3-node stencil central difference. As a result, derivatives
performed in the same direction has a stencil of width 5.

DIFFUSION_SCHEME_CENTRAL_3: the inner derivatives are computed with neighboring
nodes, so that their values fall on the faces. The outer derivatives are
performed to the face flux so that the diffusion terms fall back on nodes.
Interpolations across faces in different directions are required in this
approach.

DIFFUSION_SCHEME_STENCIL_3: the inner derivatives are computed with the 3-node
stencil central difference, except when the outer derivative is in the same
direction as the inner one, in which case both derivatives are computed from
neighboring nodes/faces. In this approach the width of the stencil in each
direction is 3.
"""


from typing import Callable, Optional

import jax
import jax.numpy as jnp
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import monin_obukhov_similarity_theory as most_lib
from swirl_lm.jax.equations import common
from swirl_lm.jax.equations import utils as eq_utils
from swirl_lm.jax.numerics import derivatives
from swirl_lm.jax.numerics import interpolation
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import types
from swirl_lm.numerics import numerics_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

_AXES = ('x', 'y', 'z')


def diffusion_scalar(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    deriv_lib: derivatives.Derivatives,
    phi: ScalarField,
    rho: ScalarField,
    diffusivity: ScalarField,
    helper_variables: Optional[ScalarFieldMap] = None,
) -> list[ScalarField]:
  """Computes the diffusion term for the conservative scalar.

  Computes [d/dx(rho*D*dphi/dx), d/dy(rho*D*dphi/dy), d/dz(rho*D*dphi/dz)].

  Note: This is the simple version without MOST or prescribed flux BCs. For
  full support, use `diffusion_scalar_factory`.

  Args:
    kernel_op: An object holding a library of kernel operations.
    deriv_lib: An instance of the derivatives library.
    phi: The scalar for which the diffusion term is computed.
    rho: The density of the fluid.
    diffusivity: The kinematic diffusivity of the scalar.
    helper_variables: A dictionary that stores variables that provides
      additional information for computing the diffusion term, e.g. scale
      factors for stretched grids.

  Returns:
    A list that contains the 3 diffusion components of the scalar.
  """
  if helper_variables is None:
    helper_variables = {}

  rho_d = rho * diffusivity

  # Compute diffusive fluxes for each dimension, evaluated on faces, so that
  # fluxes_face = [rho*D dphi/dx, rho*D dphi/dy, rho*D dphi/dz].
  fluxes_face = []
  for axis in _AXES:
    # Interpolate rho*D onto faces in dimension `axis`.
    rho_d_face = interpolation.centered_node_to_face(rho_d, axis, kernel_op)
    # Compute dphi/dx_j in dimension `axis` on faces.
    dphi_face = deriv_lib.deriv_node_to_face(phi, axis, helper_variables)
    # Compute diffusive fluxes rho*D dphi/dx_j evaluated on faces.
    flux_face = rho_d_face * dphi_face
    fluxes_face.append(flux_face)

  # Compute diffusion_terms = [d/dx(rho*D dphi/dx), d/dy(rho*D dphi/dy),
  #                            d/dz(rho*D dphi/dz)] evaluated on nodes.
  diffusion_terms = [
      deriv_lib.deriv_face_to_node(fluxes_face[i], axis, helper_variables)
      for i, axis in enumerate(_AXES)
  ]

  return diffusion_terms


def diffusion_scalar_factory(
    params: parameters_lib.SwirlLMParameters,
) -> Callable[..., list[ScalarField]]:
  """Creates a scalar diffusion function with MOST and prescribed flux BCs.

  When the MOST boundary model is configured, the surface diffusive flux is
  replaced with the MOST closure for active scalars. Prescribed diffusive
  fluxes from the proto config override all other flux values.

  Args:
    params: The simulation parameter context.

  Returns:
    A function that computes the diffusion terms for a scalar transport
    equation, with MOST and prescribed flux BC support.
  """
  if params.boundary_models is not None and params.boundary_models.HasField(
      'most'
  ):
    most = most_lib.monin_obukhov_similarity_theory_factory(params)
  else:
    most = None

  def diffusion_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      deriv_lib: derivatives.Derivatives,
      phi: ScalarField,
      rho: ScalarField,
      diffusivity: ScalarField,
      scalar_name: str | None = None,
      helper_variables: ScalarFieldMap | None = None,
  ) -> list[ScalarField]:
    """Computes the diffusion term with MOST and prescribed flux BCs.

    Args:
      kernel_op: An object holding a library of kernel operations.
      deriv_lib: An instance of the derivatives library.
      phi: The scalar for which the diffusion term is computed.
      rho: The density of the fluid.
      diffusivity: The kinematic diffusivity of the scalar.
      scalar_name: The name of the scalar. Used for MOST active scalar check and
        prescribed flux BC lookup.
      helper_variables: A dictionary that stores helper variables, including
        velocity and potential temperature for MOST.

    Returns:
      A list that contains the 3 diffusion components of the scalar.
    """
    if helper_variables is None:
      helper_variables = {}

    rho_d = rho * diffusivity

    # Compute diffusive fluxes for each dimension on faces.
    fluxes_face = []
    for axis in _AXES:
      rho_d_face = interpolation.centered_node_to_face(rho_d, axis, kernel_op)
      dphi_face = deriv_lib.deriv_node_to_face(phi, axis, helper_variables)
      flux_face = rho_d_face * dphi_face
      fluxes_face.append(flux_face)

    # Add MOST scalar flux closure at the ground surface.
    if (
        most is not None
        and scalar_name is not None
        and most.is_active_scalar(scalar_name)
    ):
      required_variables = ('u', 'v', 'w', 'theta')
      for varname in required_variables:
        if varname not in helper_variables:
          raise ValueError(f'{varname} is missing for the MOST model.')

      scalar_flux_vars: dict[str, ScalarField] = {
          'rho': rho,
          'phi': phi,
      }
      scalar_flux_vars.update(helper_variables)
      q_3 = most.surface_flux_update_fn(scalar_flux_vars, scalar_name)

      # Reverse sign for consistency with the diffusion scheme convention.
      q_3 = -q_3

      # Replace the ground-level plane in the vertical flux.
      g_dim = most.vertical_dim
      g_axis = _AXES[g_dim]
      axis_index: int = params.grid_params.get_axis_index(g_axis)  # pyrefly: ignore[bad-assignment]
      plane = jnp.expand_dims(q_3, axis=axis_index)
      start_idx = [0, 0, 0]
      start_idx[axis_index] = params.halo_width
      fluxes_face[g_dim] = jax.lax.dynamic_update_slice(
          fluxes_face[g_dim], plane, tuple(start_idx)
      )

    # Apply prescribed diffusive fluxes from proto config.
    if scalar_name is not None and scalar_name in params.scalar_lib:
      for flux_info in params.scalar_lib[scalar_name].diffusive_flux:
        dim = flux_info.dim
        axis = _AXES[dim]
        axis_index_dim: int = params.grid_params.get_axis_index(axis)  # pyrefly: ignore[bad-assignment]
        face = flux_info.face

        if flux_info.WhichOneof('flux') == 'value':
          # Constant flux value — build a full plane.
          plane_shape = list(fluxes_face[dim].shape)
          plane_shape[axis_index_dim] = 1
          flux_plane = jnp.full(plane_shape, flux_info.value)
        else:
          # Variable-based flux from helper_variables.
          flux_var = helper_variables[flux_info.varname]
          if flux_var.ndim == 2:
            flux_plane = jnp.expand_dims(flux_var, axis=axis_index_dim)
          else:
            flux_plane = flux_var

        # Compute the index for the specified face.
        n = fluxes_face[dim].shape[axis_index_dim]
        if face == 0:
          plane_idx = params.halo_width
        else:
          plane_idx = n - params.halo_width

        start = [0, 0, 0]
        start[axis_index_dim] = plane_idx
        fluxes_face[dim] = jax.lax.dynamic_update_slice(
            fluxes_face[dim], flux_plane, tuple(start)
        )

    # Compute diffusion terms on nodes.
    diffusion_terms = [
        deriv_lib.deriv_face_to_node(fluxes_face[i], axis, helper_variables)
        for i, axis in enumerate(_AXES)
    ]

    return diffusion_terms

  return diffusion_fn


def _diffusion_momentum_stencil_3(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    mu: ScalarField,
    grid_spacing: tuple[float, float, float],
    velocity: dict[str, ScalarField],
) -> dict[str, list[ScalarField]]:
  """Computes diffusion terms of momentum equations with 3-node stencil.

  Args:
    kernel_op: An object holding a library of kernel operations.
    mu: The dynamic viscosity.
    grid_spacing: A tuple that holds (dx, dy, dz).
    velocity: A dictionary that has flow field variables u, v, and w.

  Returns:
    A dictionary that holds the diffusion terms in all momentum equations. The
    dictionary is indexed by the name of the velocity components, i.e. 'u', 'v',
    and 'w'. For each velocity component, the 3 diffusion terms are stored in a
    list of 3 elements, with the elements being the diffusion component in the
    x, y, and z directions, respectively.
  """
  keys_velocity = ('u', 'v', 'w')

  # Prepares the scaled/unscaled viscosity on faces.
  mu_dim = [
      interpolation.centered_node_to_face(mu, axis, kernel_op) for axis in _AXES
  ]
  four_thirds_mu = [4.0 / 3.0 * mu_dim[i] for i in range(3)]
  two_thirds_mu = 2.0 / 3.0 * mu

  # Computes velocity gradients on faces along all directions.
  flux_u = {
      k: [
          kernel_op.apply_kernel_op(velocity[k], 'kd', axis) / grid_spacing[i]
          for i, axis in enumerate(_AXES)
      ]
      for k in keys_velocity
  }
  # Computes velocity gradients with central difference.
  grad_central_u = {
      k: [
          kernel_op.apply_kernel_op(velocity[k], 'kD', axis)
          / (2.0 * grid_spacing[i])
          for i, axis in enumerate(_AXES)
      ]
      for k in keys_velocity
  }

  def tangential_diffusion_fn(dim: int) -> ScalarField:
    """Computes the diffusion term along direction of a velocity component."""
    dims_n = [0, 1, 2]
    dims_n.remove(dim)
    axis = _AXES[dim]

    four_thirds_mu_flux_u = (
        four_thirds_mu[dim] * flux_u[keys_velocity[dim]][dim]
    )
    output = (
        kernel_op.apply_kernel_op(four_thirds_mu_flux_u, 'kd+', axis)
        / grid_spacing[dim]
    )
    for i in dims_n:
      two_thirds_mu_grad_central_u = (
          two_thirds_mu * grad_central_u[keys_velocity[i]][i]
      )
      buf = kernel_op.apply_kernel_op(
          two_thirds_mu_grad_central_u, 'kD', axis
      ) / (2 * grid_spacing[dim])
      output = output - buf

    return output

  def normal_diffusion_fn(dim: int, dim_n: int) -> ScalarField:
    """Computes the diffusion term normal to a velocity component."""
    axis_n = _AXES[dim_n]
    mu_flux_u = mu_dim[dim_n] * flux_u[keys_velocity[dim]][dim_n]
    mu_grad_central_u = mu * grad_central_u[keys_velocity[dim_n]][dim]
    dx = grid_spacing[dim_n]
    return kernel_op.apply_kernel_op(
        mu_flux_u, 'kd+', axis_n
    ) / dx + kernel_op.apply_kernel_op(mu_grad_central_u, 'kD', axis_n) / (
        2 * dx
    )

  def diffusion_fn(vel: str) -> list[ScalarField]:
    """Computes the diffusion terms of velocity component `vel`."""
    vel_id = keys_velocity.index(vel)

    output = []
    for i in range(3):
      if i == vel_id:
        output.append(tangential_diffusion_fn(vel_id))
      else:
        output.append(normal_diffusion_fn(vel_id, i))

    return output

  return {k: diffusion_fn(k) for k in keys_velocity}


def diffusion_momentum(
    params: parameters_lib.SwirlLMParameters,
) -> Callable[..., dict[str, list[ScalarField]]]:
  """Generates a function that computes the momentum diffusion terms.

  This factory creates a closure that dispatches to the appropriate diffusion
  scheme based on the ``scheme`` argument:

  - DIFFUSION_SCHEME_CENTRAL_5: Uses ``shear_stress`` (centered derivatives) and
    then applies centered derivatives of the stress tensor.
  - DIFFUSION_SCHEME_CENTRAL_3: Uses ``shear_flux`` (face-based derivatives) and
    then applies face-to-node derivatives.
  - DIFFUSION_SCHEME_STENCIL_3: Uses the 3-node stencil approach with
    tangential/normal decomposition.

  Args:
    params: The simulation parameter context.

  Returns:
    A function that computes the diffusion terms in the momentum equation.
  """
  shear_flux_fn_stencil_3 = eq_utils.shear_flux(params)

  def diffusion_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      deriv_lib: derivatives.Derivatives,
      scheme: 'numerics_pb2.DiffusionScheme',
      mu: ScalarField,
      grid_spacing: tuple[float, float, float],
      states: ScalarFieldMap,
      helper_variables: ScalarFieldMap,
      tau_bc_update_fn: (
          dict[str, Callable[[ScalarField], ScalarField]] | None
      ) = None,
  ) -> dict[str, list[ScalarField]]:
    """Computes the diffusion term in momentum equations of u, v, and w.

    Args:
      kernel_op: An object holding a library of kernel operations.
      deriv_lib: An instance of the derivatives library.
      scheme: The numerical scheme used to compute the diffusion term.
      mu: The dynamic viscosity.
      grid_spacing: A tuple that holds (dx, dy, dz).
      states: A dictionary that has flow field variables u, v, w, and rho.
      helper_variables: A dictionary that stores variables that provides
        additional information for computing the diffusion term, e.g. the
        potential temperature for the Monin-Obukhov similarity theory.
      tau_bc_update_fn: A dictionary of halo_exchange functions for the shear
        stress tensor.

    Returns:
      A dictionary that holds the diffusion terms in all momentum equations. The
      dictionary is indexed by the name of the velocity components, i.e. 'u',
      'v', and 'w'. For each velocity component, the 3 diffusion terms are
      stored in a list of 3 elements, with the elements being the diffusion
      component in the x, y, and z directions, respectively.
    """
    shear_key = {
        'u': ('xx', 'xy', 'xz'),
        'v': ('yx', 'yy', 'yz'),
        'w': ('zx', 'zy', 'zz'),
    }

    if scheme == numerics_pb2.DiffusionScheme.DIFFUSION_SCHEME_CENTRAL_5:
      tau = eq_utils.shear_stress(
          deriv_lib,
          mu,
          states['u'],
          states['v'],
          states['w'],
          helper_variables,
          tau_bc_update_fn,
      )

      def diffusion_fn_1d(
          key: str,
          dim: int,
      ) -> ScalarField:
        """Computes the diffusion term for `key` in direction `dim`."""
        shear = tau[shear_key[key][dim]]
        return deriv_lib.deriv_centered(shear, _AXES[dim], helper_variables)

      return {
          key: [diffusion_fn_1d(key, i) for i in range(3)]
          for key in common.KEYS_VELOCITY
      }
    elif scheme == numerics_pb2.DiffusionScheme.DIFFUSION_SCHEME_CENTRAL_3:
      # Compute the stress tensor tau_ij, evaluated on faces in dim j.
      tau = shear_flux_fn_stencil_3(
          kernel_op,
          deriv_lib,
          mu,
          states['u'],
          states['v'],
          states['w'],
          states[common.KEY_RHO],
          helper_variables,
      )

      def tau_deriv(key: str, dim: int) -> ScalarField:
        """Computes d(tau_ij)/dx_j (j=`dim`) with the result on nodes."""
        tau_ij = tau[shear_key[key][dim]]
        return deriv_lib.deriv_face_to_node(
            tau_ij, _AXES[dim], helper_variables
        )

      return dict(
          u=[tau_deriv('u', 0), tau_deriv('u', 1), tau_deriv('u', 2)],
          v=[tau_deriv('v', 0), tau_deriv('v', 1), tau_deriv('v', 2)],
          w=[tau_deriv('w', 0), tau_deriv('w', 1), tau_deriv('w', 2)],
      )
    elif scheme == numerics_pb2.DiffusionScheme.DIFFUSION_SCHEME_STENCIL_3:
      return _diffusion_momentum_stencil_3(
          kernel_op, mu, grid_spacing, dict(states)
      )
    else:
      raise ValueError(
          f'{scheme} is not implemented. Available options are: '
          '"DIFFUSION_SCHEME_CENTRAL_3", '
          '"DIFFUSION_SCHEME_CENTRAL_5", '
          '"DIFFUSION_SCHEME_STENCIL_3".'
      )

  return diffusion_fn


# Keep the public alias for backward compatibility with existing call sites.
diffusion_momentum_stencil_3 = _diffusion_momentum_stencil_3
