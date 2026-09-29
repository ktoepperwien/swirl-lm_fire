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
"""A library for filter operators.

This is the JAX port of `swirl_lm.numerics.filters`. All operations work on
3D `jax.Array` directly (no list-of-2D-slices support).
"""


import functools
from typing import Callable

import jax
import jax.numpy as jnp
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization as grid_parametrization_lib
from swirl_lm.jax.utility import stretched_grid_util
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


def filter_op(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    grid_params: grid_parametrization_lib.GridParametrization,
    f: ScalarField,
    additional_states: ScalarFieldMap,
    order: int = 2,
) -> ScalarField:
  """Performs filtering to a variable.

  Args:
    kernel_op: The kernel operator for finite differences.
    grid_params: The grid parametrization.
    f: The 3D variable to be filtered. Values in halos must be valid.
    additional_states: Mapping that contains the optional scale factors.
    order: The harmonic order of the filter.

  Returns:
    The filtered variable f, with values in the outermost layer being
    unfiltered.

  Raises:
    NotImplementedError: If order is not 2.
  """
  if order == 2:
    return filter_2(kernel_op, grid_params, f, additional_states)
  else:
    raise NotImplementedError(
        f'Order {order} filter is not supported. Available orders: 2.'
    )


def filter_2(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    grid_params: grid_parametrization_lib.GridParametrization,
    f: ScalarField,
    additional_states: ScalarFieldMap,
    stencil: int = 27,
) -> ScalarField:
  r"""Performs filtering by adding second-order derivatives to a variable.

  For stencil = 7:
  \bar{f}_ijk = (1 - s)f_ijk + s/6(f_{i-1,j,k} + f_{i+1,j,k} +
      f_{i,j-1,k} + f_{i,j+1,k} + f_{i,j,k-1} + f_{i,j,k+1})
  For s = 0.5,
  \bar{f}_ijk = f_ijk + 1/12(\nabla^2_x f_ijk + \nabla^2_y f_ijk +
      \nabla^2_z f_ijk)
  For stencil = 27:
  \bar{f}_i = f_i + 1/4\nabla^2_x f_i
  \bar{f}_ij = \bar{f}_i + 1/4\nabla^2_y \bar{f}_i
  \bar{f}_ijk = \bar{f}_ij + 1/4\nabla^2_z \bar{f}_ij

  Reference:
  Haltiner, George J., and Roger Terry Williams. Numerical prediction and
  dynamic meteorology. No. 551.5 HAL. 1980, p. 392-397.

  Args:
    kernel_op: The kernel operator for finite differences.
    grid_params: The grid parametrization.
    f: The 3D variable to be filtered. Values in halos must be valid.
    additional_states: Mapping that contains the optional scale factors.
    stencil: The width of the stencil to be considered for filtering. Only 7 and
      27 are allowed.

  Returns:
    The filtered variable f, with values in the outermost layer being
    unfiltered.

  Raises:
    ValueError: If stencil is not 7 or 27.
  """
  kernel_op.add_kernel(
      {'shift_up': ([1.0, 0.0, 0.0], 1), 'shift_dn': ([0.0, 0.0, 1.0], 1)}
  )

  axes = grid_params.data_axis_order
  shift_up_ops = tuple(
      functools.partial(kernel_op.apply_kernel_op, name='shift_up', axis=ax)
      for ax in axes
  )
  shift_dn_ops = tuple(
      functools.partial(kernel_op.apply_kernel_op, name='shift_dn', axis=ax)
      for ax in axes
  )
  filter_ops = tuple(
      functools.partial(kernel_op.apply_kernel_op, name='kdd', axis=ax)
      for ax in axes
  )

  use_stretched_grid = grid_params.use_stretched_grid
  g = f

  for dim in range(3):
    if use_stretched_grid[dim]:
      # Approximate coefficients of the tophat filter with the Simpson's rule.
      physical_dim = ('x', 'y', 'z').index(axes[dim])
      h_0 = additional_states[stretched_grid_util.h_key(physical_dim)]
      h_1 = shift_dn_ops[dim](h_0)
      w_0 = (0.5 * (h_0 - h_1) - (h_0 - h_1) ** 2 / (6.0 * h_0) + h_1 / 3.0) / (
          h_0 + h_1
      )
      w_1 = 2.0 / 3.0 + (h_0 - h_1) ** 2 / (6.0 * h_0 * h_1)
      w_2 = (
          -0.5 * (h_0 - h_1) - (h_0 - h_1) ** 2 / (6.0 * h_1) + h_0 / 3.0
      ) / (h_0 + h_1)
      g = w_0 * shift_up_ops[dim](g) + w_1 * g + w_2 * shift_dn_ops[dim](g)
    elif stencil == 7:
      g = g + (1.0 / 12.0) * filter_ops[dim](f)
    elif stencil == 27:
      g = g + 0.25 * filter_ops[dim](g)
    else:
      raise ValueError(
          f'Stencil width {stencil} is not supported. Allowed stencil '
          'widths are 7 and 27.'
      )

  # Preserve original values at the outermost layer (boundary).
  n0, n1, n2 = f.shape
  mask = jnp.pad(
      jnp.ones((n0 - 2, n1 - 2, n2 - 2), dtype=jnp.bool_),
      pad_width=((1, 1), (1, 1), (1, 1)),
      mode='constant',
      constant_values=False,
  )
  return jnp.where(mask, g, f)


def global_box_filter_3d(
    state: ScalarField,
    halo_update_fn: Callable[[ScalarField], ScalarField],
    kernel_op: get_kernel_fn.ApplyKernelOp,
    filter_width: int,
    num_iter: int,
) -> ScalarField:
  """Applies a balanced 3D Tophat filter to a 3D tensor.

  The following operation is performed by this function:
    u'_{lmn} =
      sum_{p=l-N/2}^{l+N/2} sum_{s=m-N/2}^{m+N/2} sum_{t=n-N/2}^{n+N/2}
        1/N^3 u_pst
  Note that the filter is balanced, so only odd number is allowed as
  `filter_width`.

  Args:
    state: The 3D tensor field to be filtered.
    halo_update_fn: A function that is used to update the halos of `f`.
    kernel_op: The kernel operator to use for filtering. Must support the
      `apply_kernel_op` method with axis specification.
    filter_width: The full width of stencil of the filter in each direction.
    num_iter: The number of iterations that the filter is applied.

  Returns:
    The filtered `state`.

  Raises:
    ValueError: If `filter_width` is even.
  """
  if filter_width % 2 == 0:
    raise ValueError(
        f'Filter width has to be an odd number. {filter_width} is not allowed.'
    )

  filter_coeffs = [1.0 / filter_width] * filter_width
  offset = filter_width // 2
  kernel_dict = {'filter': (filter_coeffs, offset)}
  kernel_op.add_kernel(kernel_dict)

  def body(carry: tuple[int, ScalarField]) -> tuple[int, ScalarField]:
    i, f_filtered = carry
    filtered_x = kernel_op.apply_kernel_op(f_filtered, 'filter', 'x')
    filtered_y = kernel_op.apply_kernel_op(filtered_x, 'filter', 'y')
    filtered_z = kernel_op.apply_kernel_op(filtered_y, 'filter', 'z')
    return (i + 1, halo_update_fn(filtered_z))

  def cond(carry: tuple[int, ScalarField]) -> bool:
    i, _ = carry
    return i < num_iter

  _, f_filtered = jax.lax.while_loop(cond, body, (0, state))
  return f_filtered
