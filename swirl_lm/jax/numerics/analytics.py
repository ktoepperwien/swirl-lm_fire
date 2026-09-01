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
"""Library for computing basic analytics (JAX port).

This is the JAX port of `swirl_lm.numerics.analytics`. In single-device mode,
global reductions simplify to standard jnp operations over the full array.
"""


from collections.abc import Sequence
from typing import Optional

import jax.numpy as jnp
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField


def _strip_halos(f: ScalarField, halos: Sequence[int]) -> ScalarField:
  """Strips halo regions from a 3D field.

  Args:
    f: A 3D array.
    halos: Halo widths for each dimension.

  Returns:
    The field with halos stripped.
  """
  slices = tuple(slice(h, -h) if h > 0 else slice(None) for h in halos)
  return f[slices]


def _clear_halos(f: ScalarField, halo_width: int) -> ScalarField:
  """Zeros out the halo regions of a 3D field.

  Args:
    f: A 3D array.
    halo_width: Width of the halo to zero.

  Returns:
    The field with halo regions set to zero.
  """
  if halo_width <= 0:
    return f
  mask = jnp.ones_like(f)
  hw = halo_width
  # Zero out halos in all three dimensions.
  mask = mask.at[:hw, :, :].set(0.0)
  mask = mask.at[-hw:, :, :].set(0.0)
  mask = mask.at[:, :hw, :].set(0.0)
  mask = mask.at[:, -hw:, :].set(0.0)
  mask = mask.at[:, :, :hw].set(0.0)
  mask = mask.at[:, :, -hw:].set(0.0)
  return f * mask


def moments(
    f1: ScalarField,
    n: Sequence[int],
    halos: Sequence[int],
    homogeneous_dims: Sequence[bool],
    f2: Optional[ScalarField] = None,
    f1_ref: Optional[ScalarField] = None,
    f2_ref: Optional[ScalarField] = None,
) -> list[ScalarField]:
  """Calculates moments of the field: E[(f - E[f])^k].

  The calculation will exclude the halo region, and there will be one moment
  calculated for each of the exponents contained in `n`. The moments are
  calculated relative to the mean in the homogeneous directions, or relative to
  a reference state if one is provided.

  Args:
    f1: The field/variable as a 3D array.
    n: A sequence of the moment orders to compute. Should always be >= 1.
    halos: The width of the (symmetric) halos for each dimension.
    homogeneous_dims: A sequence of booleans indicating the homogeneous
      dimensions of the physical grid.
    f2: Optional second field for computing a cross moment with `f1`.
    f1_ref: Optional reference to subtract from `f1` before computing moments.
      If not provided, the mean of `f1` is used.
    f2_ref: Optional reference to subtract from `f2`. If not provided, the mean
      of `f2` is used.

  Returns:
    A list of arrays representing the n-th order moment of the field(s).
  """
  reduction_axis = tuple(
      i for i, active in enumerate(homogeneous_dims) if active
  )

  # Strip halos from all fields.
  f1 = _strip_halos(f1, halos)
  f1_mean = jnp.mean(f1, axis=reduction_axis, keepdims=True)

  if f1_ref is not None:
    f1_ref = _strip_halos(f1_ref, halos)
  else:
    f1_ref = f1_mean

  if f2 is not None:
    f2 = _strip_halos(f2, halos)
    if f2_ref is not None:
      f2_ref = _strip_halos(f2_ref, halos)
    else:
      f2_ref = jnp.mean(f2, axis=reduction_axis, keepdims=True)
    f2 = f2 - f2_ref

  moment_lst = []
  for order in n:
    if f2 is None:
      if order == 1:
        moment_lst.append(f1_mean)
      else:
        deviation = f1 - f1_ref
        moment_n = jnp.power(deviation, order)
        moment_lst.append(moment_n)
    else:  # Compute cross moment.
      deviation = f1 - f1_ref
      f_cross = deviation * f2
      moment_n = jnp.power(f_cross, order)
      moment_lst.append(moment_n)

  return [
      jnp.mean(moment, axis=reduction_axis, keepdims=True)
      for moment in moment_lst
  ]


def pair_distance_with_tol(
    lhs: ScalarField,
    rhs: ScalarField,
    atol: float,
    rtol: float,
    halo_width: int,
    symmetric: bool = False,
) -> ScalarField:
  r"""Computes max distance of `lhs` vs `rhs` without halos, within tolerance.

  By default using the `rhs` as the reference only, but could also use the `lhs`
  as the reference at the same time if `symmetric` is set.

  1. Compute `tol = atol + rtol * abs(rhs)`
  2. Compute `diff = lhs - rhs`
  3. Define `distance = max(abs(diff) - tol)` and check:
     - Close enough if non-positive, where `-tol <= diff <= +tol`.
     - Not close if positive, where `diff > tol` or `diff < -tol`.

  Args:
    lhs: Array on the left hand side for distance.
    rhs: Array on the right hand side for distance, and used as reference for
      relative tolerance `rtol`.
    atol: Absolute difference for comparison.
    rtol: Relative difference for comparison.
    halo_width: Width of the halo.
    symmetric: Whether to use the `lhs` as reference as well.

  Returns:
    The maximum distance (defined as a max difference) of two vectors.

  Raises:
    ValueError: If `atol` or `rtol` is negative.
  """
  if atol < 0 or rtol < 0:
    raise ValueError(f'Invalid input of (atol, rtol) = ({atol}, {rtol}) < 0.')

  lhs = _clear_halos(lhs, halo_width)
  rhs = _clear_halos(rhs, halo_width)
  diff = lhs - rhs

  def pair_distance_fn(d: ScalarField, raw: ScalarField) -> ScalarField:
    """Max distance from absolute diff and raw vector."""
    tol = atol + rtol * jnp.abs(raw)
    distance = jnp.abs(d) - tol
    return jnp.max(distance)

  rhs_distance = pair_distance_fn(diff, rhs)
  if symmetric:
    return jnp.maximum(pair_distance_fn(diff, lhs), rhs_distance)
  else:
    return rhs_distance
