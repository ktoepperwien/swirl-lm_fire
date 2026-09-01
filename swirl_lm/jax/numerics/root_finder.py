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
"""Root finding methods for JAX.

Provides a Newton/secant method for finding roots of scalar-valued functions
operating on 3D arrays (ScalarField). Used by the Monin-Obukhov Similarity
Theory module for computing the normalized height (Obukhov length).

The implementation uses `jax.lax.fori_loop` for a fixed number of iterations
(no early stopping) to maintain JIT compatibility.
"""


from collections.abc import Callable

import jax
import jax.numpy as jnp
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField

# A small number used as perturbation when the solution is zero.
_EPS = 1e-4


def newton_method(
    objective_fn: Callable[[ScalarField], ScalarField],
    initial_position: ScalarField,
    max_iterations: int,
    analytical_jacobian_fn: Callable[[ScalarField], ScalarField] | None = None,
) -> ScalarField:
  """Finds the root of `objective_fn` with the Newton/secant method.

  For a scalar function f(x) = 0, the Newton iteration is:
    x_{n+1} = x_n - f(x_n) / f'(x_n)

  When `analytical_jacobian_fn` is None, the derivative is estimated using
  central finite differences (secant method).

  Args:
    objective_fn: The function whose root is sought. Takes and returns a
      ScalarField (3D JAX array).
    initial_position: Initial guess for the root.
    max_iterations: Fixed number of Newton iterations to run.
    analytical_jacobian_fn: Optional function computing the derivative of
      `objective_fn`. If None, a numerical finite-difference derivative is used.

  Returns:
    The approximate root of `objective_fn`.
  """
  if max_iterations <= 0:
    return initial_position

  # Compute machine epsilon for the perturbation in numerical derivatives.
  dtype = initial_position.dtype
  eps = jnp.finfo(dtype).resolution
  eps = jnp.power(2.0, jnp.ceil(jnp.log(10.0 * eps) / jnp.log(2.0)))

  def numerical_jacobian_fn(x: ScalarField) -> ScalarField:
    """Estimates the derivative using central finite differences."""
    dx = eps * jnp.abs(x)
    dx = jnp.where(dx == 0.0, _EPS, dx)
    x1 = x - dx / 2.0
    x2 = x + dx / 2.0
    return (objective_fn(x2) - objective_fn(x1)) / dx

  jacobian_fn = (
      numerical_jacobian_fn
      if analytical_jacobian_fn is None
      else analytical_jacobian_fn
  )

  def body_fn(_: int, x: ScalarField) -> ScalarField:
    """One Newton iteration: x <- x - f(x) / f'(x)."""
    f = objective_fn(x)
    df = jacobian_fn(x)
    # Safe division: when df == 0, the update is 0 (no change).
    h = jnp.where(df != 0.0, f / df, 0.0)
    return x - h

  return jax.lax.fori_loop(0, max_iterations, body_fn, initial_position)
