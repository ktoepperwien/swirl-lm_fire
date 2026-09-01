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
"""A library for synthetic turbulent inflow boundary conditions (JAX port).

This is the JAX port of
`swirl_lm.boundary_condition.synthetic_turbulent_inflow`.

Generates inflow with synthetic turbulence using the digital filter method of
Klein, Sadiki, and Janicka (2003).

Reference:
Klein, M., A. Sadiki, and J. Janicka. 2003. "A Digital Filter Based Generation
of Inflow Data for Spatially Developing Direct Numerical or Large Eddy
Simulations." Journal of Computational Physics 186 (2): 652-65.
"""


from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.base import physical_variable_keys_manager
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


class SyntheticTurbulentInflow:
  """A library that generates inflow with synthetic turbulence.

  To use this library in a fluid simulation, the synthetic turbulent inflow
  needs to be generated at the beginning of each time step, and fed as Dirichlet
  boundary conditions to all velocity components at the specified boundary via
  `additional_states`.

  To use this library, the following keys are required in `additional_states`:
    'bc_[u, v, w]_{inflow_dim}_{inflow_face}',
    'mean_[u, v, w]_{inflow_dim}_{inflow_face}',
    'rms_[u, v, w]_{inflow_dim}_{inflow_face}',
    'rand_[u, v, w]_{inflow_dim}_{inflow_face}',
  """

  def __init__(
      self,
      length_scale: Sequence[float],
      delta: Sequence[float],
      mesh_size: Sequence[int],
      inflow_dim: int,
      inflow_face: int,
  ):
    """Initializes fields and operators required for turbulence generation.

    Args:
      length_scale: Character length scales in three dimensions (x, y, z).
      delta: Grid sizes in three dimensions (x, y, z).
      mesh_size: Number of grid points in three dimensions (x, y, z).
      inflow_dim: The dimension along which the inflow is injected.
      inflow_face: The index of the face along `inflow_dim` where the inflow is
        imposed. 0 indicates the face with lower physical index, and 1 indicates
        the face with higher physical index.
    """
    self.inflow_face = inflow_face
    self.inflow_dim = inflow_dim
    self.inflow_plane = [dim for dim in range(3) if dim != inflow_dim]
    inflow_permute = [inflow_dim] + self.inflow_plane

    # The widths of the digital filter.
    n = np.array([int(np.ceil(l / d)) for l, d in zip(length_scale, delta)])[
        inflow_permute
    ]

    # Total number of points needs to be added for the filter width.
    self.n_pad = np.array([2 * n_i for n_i in n])

    # Number of grid points along the two dimensions in the inflow plane.
    self.m = np.array(mesh_size)[inflow_permute[1:]]

    # Total number of points required by the random fields.
    self.nr_total = [
        2 * self.n_pad[0],
        2 * self.n_pad[1] + self.m[0],
        2 * self.n_pad[2] + self.m[1],
    ]

    # Initializes the weights of the filters.
    self.b = [
        self._compute_filter_weights(n_i, n_pad_i)
        for n_i, n_pad_i in zip(n, self.n_pad)
    ]

    # Generate boundary condition keys for the specified inflow.
    helper = physical_variable_keys_manager.BoundaryConditionKeysHelper()
    self._bc_keys = [
        helper.generate_bc_key(varname, inflow_dim, inflow_face)
        for varname in ['u', 'v', 'w']
    ]

    # Derive keys for other required fields.
    self._mean_keys = [
        self.helper_key('mean', vel, inflow_dim, inflow_face)
        for vel in ['u', 'v', 'w']
    ]
    self._rms_keys = [
        self.helper_key('rms', vel, inflow_dim, inflow_face)
        for vel in ['u', 'v', 'w']
    ]
    self._rand_keys = [
        self.helper_key('rand', vel, inflow_dim, inflow_face)
        for vel in ['u', 'v', 'w']
    ]
    self._required_keys = (
        self._bc_keys + self._mean_keys + self._rms_keys + self._rand_keys
    )

  def helper_key(
      self,
      helper_type: str,
      velocity: str,
      inflow_dim: int,
      inflow_face: int,
  ) -> str:
    """Generates the key for a helper variable for the inflow generation.

    The key of the helper variable takes the following format:
      [helper_type]_[velocity]_[inflow_dim]_[inflow_face]

    Args:
      helper_type: The type of the helper variable. Should be one of 'bc',
        'mean', 'rms', 'rand'.
      velocity: The name of the velocity component. Should be one of 'u', 'v',
        'w'.
      inflow_dim: The inflow dimension. Should be one of 0, 1, or 2.
      inflow_face: The face at which the inflow is coming from. 0 or 1.

    Returns:
      The key of the helper variable.

    Raises:
      ValueError: If any argument is out of range.
    """
    if helper_type not in ['bc', 'mean', 'rms', 'rand']:
      raise ValueError(
          '`helper_type` must be "bc", "mean", "rms", or "rand". '
          f'{helper_type} is invalid.'
      )
    if velocity not in ['u', 'v', 'w']:
      raise ValueError(
          f'`velocity` must be "u", "v", or "w". {velocity} is invalid.'
      )
    if inflow_dim not in [0, 1, 2]:
      raise ValueError(
          f'`inflow_dim` must be 0, 1, or 2. {inflow_dim} is invalid.'
      )
    if inflow_face not in [0, 1]:
      raise ValueError(
          f'`inflow_face` must be 0 or 1. {inflow_face} is invalid.'
      )
    return f'{helper_type}_{velocity}_{inflow_dim}_{inflow_face}'

  def _compute_filter_weights(self, n: int, n_pad: int) -> list[float]:
    """Computes the filter weights for the turbulence flow field.

    Args:
      n: The number of mesh points required to capture the characteristic length
        scale.
      n_pad: The half width of the filter stencil.

    Returns:
      The weights of the filter with length being 2 * `n_pad`.
    """
    b_tilde = [np.exp(-np.pi * k**2 / n**2) for k in range(-n_pad, n_pad)]
    return (b_tilde / np.sum(b_tilde)).tolist()

  def _inflow_plane_to_bc(
      self,
      inflow: ScalarField,
      halo_width: int,
      grid_params: 'grid_parametrization.GridParametrization | None' = None,
  ) -> ScalarField:
    """Arranges the inflow plane to the boundary condition format.

    Args:
      inflow: A 2D array that contains the inflow information.
      halo_width: The width of the halo layers.
      grid_params: Grid parametrization for data_axis_order-aware transposition.
        If None, assumes data_axis_order=('x','y','z').

    Returns:
      A 3D array for the boundary condition.
    """
    # Expand and tile along the inflow dimension.
    plane = jnp.tile(jnp.expand_dims(inflow, 0), [halo_width + 1, 1, 1])
    # Pad the in-plane dimensions with zeros for halos.
    plane = jnp.pad(
        plane,
        pad_width=((0, 0), (halo_width, halo_width), (halo_width, halo_width)),
    )
    # `plane` has shape (inflow_size, plane_dim_0_size, plane_dim_1_size)
    # where the dimensions correspond to physical axes:
    #   [inflow_dim, inflow_plane[0], inflow_plane[1]].
    # We need to permute to data_axis_order.
    physical_order = [self.inflow_dim] + self.inflow_plane
    if grid_params is not None:
      data_order = list(grid_params.data_axis_order)
    else:
      data_order = ['x', 'y', 'z']
    axes_names = ['x', 'y', 'z']
    # physical_order[i] gives the physical axis index for plane dim i.
    # We need perm such that plane dim perm[j] goes to output dim j.
    perm = [
        physical_order.index(axes_names.index(data_order[j])) for j in range(3)
    ]
    return jnp.transpose(plane, perm)

  def generate_random_fields(
      self,
      key: jax.Array,
  ) -> list[ScalarField]:
    """Generates three random fields for the turbulence generation.

    Args:
      key: A JAX PRNG key.

    Returns:
      A length 3 list of 3D arrays, each being a random field.
    """
    keys = jax.random.split(key, 3)
    return [jax.random.normal(keys[i], shape=self.nr_total) for i in range(3)]

  def compute_inflow_velocity(
      self,
      r: list[ScalarField],
      velocity_mean: list[ScalarField],
      velocity_rms: list[ScalarField],
      key: jax.Array,
  ) -> dict[str, list[ScalarField]]:
    """Computes the inflow velocity with synthetic turbulence.

    In the single-device JAX port, halo exchange is a no-op (the full domain
    is on one device), so we skip it and work directly with the random fields.

    Args:
      r: A 3 element list of 3D arrays, each being a random field.
      velocity_mean: The mean profile for velocity in three dimensions. Each
        velocity component is a 2D array covering the inflow plane.
      velocity_rms: The rms profile for velocity in three dimensions. Each
        component is a 2D array covering the inflow plane.
      key: A JAX PRNG key for generating new random values.

    Returns:
      A dictionary with 'r' (updated random fields) and 'u' (inflow velocity
      components as 2D arrays).

    Raises:
      ValueError: If shapes of inputs are incompatible.
    """

    def compute_u_alpha(r_alpha: ScalarField) -> ScalarField:
      """Computes the alpha component of u via digital filtering."""
      filter_dim_1 = np.sum(
          [
              np.eye(
                  M=(self.m[1] + 2 * self.n_pad[2]),
                  N=self.m[1],
                  k=i,
              )
              * self.b[2][i]
              for i in range(2 * self.n_pad[2])
          ],
          axis=0,
      )
      filter_dim_0 = np.sum(
          [
              np.eye(
                  M=(self.m[0] + 2 * self.n_pad[1]),
                  N=self.m[0],
                  k=i,
              )
              * self.b[1][i]
              for i in range(2 * self.n_pad[1])
          ],
          axis=0,
      )
      filter_inflow_dim = jnp.array(self.b[0])

      u_alpha_xy = jnp.einsum('lk,ijk->lij', filter_dim_1, r_alpha)
      u_alpha_x = jnp.einsum('lk,ijk->lij', filter_dim_0, u_alpha_xy)
      return jnp.einsum('k,ijk->ij', filter_inflow_dim, u_alpha_x)

    # Check the shape of the input tensors.
    for i in range(3):
      if list(r[i].shape) != self.nr_total:
        raise ValueError(
            f'The shape of random field {i} is not compatible. '
            f'{r[i].shape} is given but {self.nr_total} is requested.'
        )

    # In single-device mode, no halo exchange is needed.
    u_alpha = [compute_u_alpha(r_alpha) for r_alpha in r]

    # Shift random fields and add new random slice at end.
    keys = jax.random.split(key, 3)
    r_new = []
    for i in range(3):
      new_slice = jax.random.normal(
          keys[i],
          shape=[
              self.n_pad[1] * 2 + self.m[0],
              self.n_pad[2] * 2 + self.m[1],
          ],
      )
      r_new.append(
          jnp.concatenate(
              [r[i][1:, ...], jnp.expand_dims(new_slice, axis=0)],
              axis=0,
          )
      )

    # u = mean + rms * u_alpha
    u = [velocity_mean[i] + velocity_rms[i] * u_alpha[i] for i in range(3)]

    return {'r': r_new, 'u': u}

  def generate_inflow_update_fn(
      self,
      key: Optional[jax.Array] = None,
  ):
    """Generates an additional_states update function for inflow.

    Args:
      key: An optional JAX PRNG key. If None, a default key is created.

    Returns:
      A function that updates additional_states with inflow BCs.
    """

    def additional_states_update_fn(
        states: ScalarFieldMap,
        additional_states: ScalarFieldMap,
        params: parameters_lib.SwirlLMParameters,
    ) -> dict[str, ScalarField]:
      """Updates the inflow boundary condition with synthetic turbulence."""
      del states

      for req_key in self._required_keys:
        if req_key not in additional_states:
          raise ValueError(
              f'{req_key} is required by the synthetic turbulent inflow '
              'but was not found.'
          )

      # Use the provided key or fold the step count for reproducibility.
      rng_key = key if key is not None else jax.random.PRNGKey(42)

      inflow_info = self.compute_inflow_velocity(
          [additional_states[k] for k in self._rand_keys],
          [additional_states[k] for k in self._mean_keys],
          [additional_states[k] for k in self._rms_keys],
          rng_key,
      )

      additional_states_updated: dict[str, ScalarField] = {}
      for as_key, value in additional_states.items():
        if as_key in self._rand_keys:
          additional_states_updated[as_key] = inflow_info['r'][
              self._rand_keys.index(as_key)
          ]
        elif as_key in self._bc_keys:
          additional_states_updated[as_key] = self._inflow_plane_to_bc(
              inflow_info['u'][self._bc_keys.index(as_key)],
              params.halo_width,
              params.grid_params,
          )
        else:
          additional_states_updated[as_key] = value

      return additional_states_updated

    return additional_states_update_fn
