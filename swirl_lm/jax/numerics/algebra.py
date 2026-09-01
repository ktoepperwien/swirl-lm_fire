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
"""A library for common linear algebra operations (JAX port).

This is the JAX port of `swirl_lm.numerics.algebra`. All operations are
elementwise across 3D arrays (one 2x2 or 3x3 matrix per grid cell).
"""


from collections.abc import Sequence

import jax.numpy as jnp
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField

# A matrix of ScalarFields (each row is a sequence of ScalarFields).
FieldMatrix = Sequence[Sequence[ScalarField]]


def _validate_matrix_shape(matrix: FieldMatrix, shape: tuple[int, int]) -> None:
  """Validates field matrix shape is the same as expected."""
  if len(matrix) != shape[0]:
    raise ValueError(
        f'Invalid matrix row len = {len(matrix)}: '
        f'not consistent with expected shape: {shape}.'
    )
  for m in matrix:
    if len(m) != shape[1]:
      raise ValueError(
          f'Invalid matrix col len = {len(m)}: '
          f'not consistent with expected shape: {shape}.'
      )


def det_2x2(matrix: FieldMatrix) -> ScalarField:
  r"""Computes determinant for 2x2 matrices elementwise in 3D arrays.

  M is a matrix of 3D arrays:
  M = (a, b)
      (c, d)
  where {a, b, c, d} are all 3D arrays.

  det(M) = a * d - b * c

  Args:
    matrix: The 2x2 matrix to get elementwise determinant.

  Returns:
    The elementwise determinant of the 2x2 matrix.
  """
  _validate_matrix_shape(matrix, (2, 2))
  a, b = matrix[0]
  c, d = matrix[1]
  return a * d - b * c


def det_3x3(matrix: FieldMatrix) -> ScalarField:
  r"""Computes determinant for 3x3 matrices elementwise in 3D arrays.

  M = (a, b, c)
      (d, e, f)
      (g, h, i)

  det(M) = a * (e*i - f*h) - b * (d*i - f*g) + c * (d*h - e*g)

  Args:
    matrix: The 3x3 matrix to get elementwise determinant.

  Returns:
    The elementwise determinant of the 3x3 matrix.
  """
  _validate_matrix_shape(matrix, (3, 3))
  a, b, c = matrix[0]
  d, e, f = matrix[1]
  g, h, i = matrix[2]
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _divide_no_nan(x: ScalarField, y: ScalarField) -> ScalarField:
  """Divides x by y, returning 0 where y is 0."""
  return jnp.where(y == 0, 0.0, x / y)


def solve_2x2(
    matrix: FieldMatrix, rhs: Sequence[ScalarField]
) -> list[ScalarField]:
  """Solves a linear system of 2x2 for each cell in 3D arrays.

  M * x = rhs elementwise:
    (a, b) * (x0)   (e)
    (c, d)   (x1) = (f)
  When the matrices are singular, the solution is set to zero.

  Args:
    matrix: The 2x2 matrix as a linear operator elementwise.
    rhs: The vector on the right hand side.

  Returns:
    The solution x so that M * x = rhs holds elementwise.
  """
  _validate_matrix_shape(matrix, (2, 2))
  e, f = rhs
  a, b = matrix[0]
  c, d = matrix[1]

  inv_factor = det_2x2(matrix)

  return [
      _divide_no_nan(det_2x2([[e, b], [f, d]]), inv_factor),
      _divide_no_nan(det_2x2([[a, e], [c, f]]), inv_factor),
  ]


def solve_3x3(
    matrix: FieldMatrix, rhs: Sequence[ScalarField]
) -> list[ScalarField]:
  """Solves a linear system of 3x3 for each cell in 3D arrays.

  M * x = rhs elementwise:
    (a, b, c)    (x0)   (j)
    (d, e, f)  * (x1) = (k)
    (g, h, i)    (x2)   (l)
  When the matrices are singular, the solution is set to zero.

  Args:
    matrix: The 3x3 matrix as a linear operator elementwise.
    rhs: The vector on the right hand side.

  Returns:
    The solution x so that M * x = rhs holds elementwise.
  """
  _validate_matrix_shape(matrix, (3, 3))
  a, b, c = matrix[0]
  d, e, f = matrix[1]
  g, h, i = matrix[2]
  j, k, l = rhs

  inv_factor = det_3x3(matrix)

  return [
      _divide_no_nan(det_3x3([[j, b, c], [k, e, f], [l, h, i]]), inv_factor),
      _divide_no_nan(det_3x3([[a, j, c], [d, k, f], [g, l, i]]), inv_factor),
      _divide_no_nan(det_3x3([[a, b, j], [d, e, k], [g, h, l]]), inv_factor),
  ]
