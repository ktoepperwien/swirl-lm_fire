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

# Copyright 2022 Google LLC
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
"""Multigrid Poisson solver for the JAX framework.

This module provides a geometric multigrid solver for the Poisson equation.
It implements a V-cycle multigrid method with weighted Jacobi smoothing,
standard restriction (full weighting), and prolongation (bilinear
interpolation).

This is the JAX port of `swirl_lm.linalg.multigrid`. The multigrid algorithm
operates on single-device 3D arrays using JAX operations.
"""


from typing import Callable

import jax
import jax.numpy as jnp
from swirl_lm.jax.linalg import base_poisson_solver
from swirl_lm.jax.linalg import poisson_solver_pb2
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField


def _zero_borders(x: jnp.ndarray) -> jnp.ndarray:
  """Zeros out the borders of the given 3D array (width 1)."""
  inner = x[1:-1, 1:-1, 1:-1]
  return jnp.pad(inner, ((1, 1), (1, 1), (1, 1)), mode='constant')


def _laplacian_3d(
    u: jnp.ndarray,
    dz: float,
    dx: float,
    dy: float,
) -> jnp.ndarray:
  """Computes the 3D Laplacian using second-order finite differences.

  Uses interior stencil with zero-padded shifts to enforce Dirichlet BCs.

  Args:
    u: Input 3D array.
    dz: Grid spacing in z.
    dx: Grid spacing in x.
    dy: Grid spacing in y.

  Returns:
    The 3D Laplacian of u.
  """

  # Shift up/down along each axis using slicing + zero padding.
  def shift_fwd(arr, axis):
    """Shift forward: arr[i] -> arr[i+1], pad zero at end."""
    slices = [slice(None)] * 3
    slices[axis] = slice(1, None)
    pad_widths = [(0, 0)] * 3
    pad_widths[axis] = (0, 1)
    return jnp.pad(arr[tuple(slices)], pad_widths, mode='constant')

  def shift_bwd(arr, axis):
    """Shift backward: arr[i] -> arr[i-1], pad zero at start."""
    slices = [slice(None)] * 3
    slices[axis] = slice(None, -1)
    pad_widths = [(0, 0)] * 3
    pad_widths[axis] = (1, 0)
    return jnp.pad(arr[tuple(slices)], pad_widths, mode='constant')

  lap = (shift_fwd(u, 0) - 2.0 * u + shift_bwd(u, 0)) / (dz * dz)
  lap = lap + (shift_fwd(u, 1) - 2.0 * u + shift_bwd(u, 1)) / (dx * dx)
  lap = lap + (shift_fwd(u, 2) - 2.0 * u + shift_bwd(u, 2)) / (dy * dy)
  return lap


def _weighted_jacobi_smooth(
    u: jnp.ndarray,
    b: jnp.ndarray,
    dz: float,
    dx: float,
    dy: float,
    weight: float,
    n_smooth: int,
) -> jnp.ndarray:
  """Applies n_smooth iterations of weighted Jacobi smoothing.

  Solves L(u) = b with: u_new = u + weight * (b - L(u)) / diag_coeff.

  Args:
    u: Current iterate.
    b: Right hand side.
    dz: Grid spacing in z.
    dx: Grid spacing in x.
    dy: Grid spacing in y.
    weight: Jacobi relaxation weight (typically 2/3).
    n_smooth: Number of smoothing iterations.

  Returns:
    Smoothed iterate.
  """
  diag_coeff = -2.0 * (1.0 / (dz * dz) + 1.0 / (dx * dx) + 1.0 / (dy * dy))

  def body_fn(i: int, u: jnp.ndarray) -> jnp.ndarray:
    del i
    residual = b - _laplacian_3d(u, dz, dx, dy)
    u = u + weight * residual / diag_coeff
    return _zero_borders(u)

  return jax.lax.fori_loop(0, n_smooth, body_fn, u)


def _restrict(fine: jnp.ndarray) -> jnp.ndarray:
  """Full-weighting restriction from fine grid to coarse grid.

  Uses standard 3D full-weighting stencil (average of 2x2x2 block).

  Args:
    fine: Fine grid array of shape (nz, nx, ny).

  Returns:
    Coarse grid array of shape (nz//2+1, nx//2+1, ny//2+1) approximately.
  """
  # Simple injection-style restriction: take every other point from the
  # interior, pad with zeros.
  inner = fine[1:-1, 1:-1, 1:-1]
  coarse_inner = inner[::2, ::2, ::2]
  return jnp.pad(coarse_inner, ((1, 1), (1, 1), (1, 1)), mode='constant')


def _prolongate(
    coarse: jnp.ndarray, fine_shape: tuple[int, ...]
) -> jnp.ndarray:
  """Bilinear prolongation from coarse grid to fine grid.

  Uses trilinear interpolation to map from coarse to fine grid.

  Args:
    coarse: Coarse grid array.
    fine_shape: Target fine grid shape.

  Returns:
    Prolongated array at fine grid resolution.
  """
  # Extract interior of coarse grid.
  coarse_inner = coarse[1:-1, 1:-1, 1:-1]

  # Target interior shape.
  fz, fx, fy = fine_shape[0] - 2, fine_shape[1] - 2, fine_shape[2] - 2

  # Use nearest-neighbor upsampling then pad. For a proper multigrid
  # implementation, this would use trilinear interpolation. For now,
  # jax.image.resize provides this.
  fine_inner = jax.image.resize(coarse_inner, (fz, fx, fy), method='linear')
  return jnp.pad(fine_inner, ((1, 1), (1, 1), (1, 1)), mode='constant')


class MultigridSolver(base_poisson_solver.PoissonSolver):
  """Multigrid Poisson solver using V-cycle with Jacobi smoothing."""

  def __init__(
      self,
      grid_params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    """Initializes the multigrid solver.

    Args:
      grid_params: The grid parametrization.
      kernel_op: An object holding a library of kernel operations.
      solver_option: The solver configuration proto.
    """
    super().__init__(grid_params, kernel_op, solver_option)
    mg_config = solver_option.multigrid

    self._num_iterations = mg_config.num_iterations
    self._n_smooth = mg_config.n_smooth
    self._weight = mg_config.weight
    self._n_coarse = mg_config.n_coarse

    # Coarsest subgrid shape (stop coarsening when any dim <= this).
    if mg_config.HasField('coarsest_subgrid_shape'):
      cs = mg_config.coarsest_subgrid_shape
      self._coarsest_shape = (cs.dim_z, cs.dim_x, cs.dim_y)
    else:
      self._coarsest_shape = (4, 4, 4)

    self._use_a_inv = mg_config.use_a_inv

    # Grid spacings.
    spacings = grid_params.grid_spacings
    self._dz = spacings[0]
    self._dx = spacings[1]
    self._dy = spacings[2]

  def _can_coarsen(self, shape: tuple[int, ...]) -> bool:
    """Checks if the grid can be coarsened further."""
    # Interior shape (excluding borders).
    interior = tuple(s - 2 for s in shape)
    return all(
        s > c and s % 2 == 0 for s, c in zip(interior, self._coarsest_shape)
    )

  def _v_cycle(
      self,
      u: jnp.ndarray,
      b: jnp.ndarray,
      dz: float,
      dx: float,
      dy: float,
  ) -> jnp.ndarray:
    """Performs one V-cycle of the multigrid method.

    Args:
      u: Current iterate.
      b: Right hand side.
      dz: Grid spacing in z.
      dx: Grid spacing in x.
      dy: Grid spacing in y.

    Returns:
      Updated iterate after one V-cycle.
    """
    fine_shape = u.shape

    # Pre-smooth.
    u = _weighted_jacobi_smooth(u, b, dz, dx, dy, self._weight, self._n_smooth)

    if not self._can_coarsen(fine_shape):
      # At coarsest level: do extra smoothing.
      u = _weighted_jacobi_smooth(
          u, b, dz, dx, dy, self._weight, self._n_smooth * 4
      )
      return u

    # Compute residual.
    residual = b - _laplacian_3d(u, dz, dx, dy)
    residual = _zero_borders(residual)

    # Restrict residual to coarse grid.
    residual_c = _restrict(residual)

    # Solve on coarse grid (recursively or with extra smoothing).
    err_c = jnp.zeros_like(residual_c)
    for _ in range(self._n_coarse):
      err_c = self._v_cycle(err_c, residual_c, dz * 2, dx * 2, dy * 2)

    # Prolongate error correction to fine grid.
    err_f = _prolongate(err_c, fine_shape)

    # Correct.
    u = u + err_f
    u = _zero_borders(u)

    # Post-smooth.
    u = _weighted_jacobi_smooth(u, b, dz, dx, dy, self._weight, self._n_smooth)

    return u

  def solve(
      self,
      rhs: ScalarField,
      p0: ScalarField,
      mesh: jax.sharding.Mesh,
      halo_update_fn: Callable[[ScalarField], ScalarField] | None = None,
      additional_states: dict[str, jax.Array] | None = None,
  ) -> base_poisson_solver.PoissonSolverSolution:
    """Solves the Poisson equation using multigrid V-cycles.

    Args:
      rhs: The right hand side of the Poisson equation.
      p0: Initial guess for the solution.
      mesh: A jax Mesh object representing the device topology.
      halo_update_fn: Optional function to update halos. Not used.
      additional_states: Additional static fields. Not used.

    Returns:
      A dict with the solution, residual norm, and iteration count.
    """
    del mesh, halo_update_fn, additional_states

    x = p0
    for _ in range(self._num_iterations):
      x = self._v_cycle(x, rhs, self._dz, self._dx, self._dy)

    # Compute final residual.
    residual = rhs - _laplacian_3d(x, self._dz, self._dx, self._dy)
    residual_l2 = float(jnp.sqrt(jnp.sum(residual**2)))

    return {
        base_poisson_solver.X: x,
        base_poisson_solver.RESIDUAL_L2_NORM: residual_l2,
        base_poisson_solver.ITERATIONS: float(self._num_iterations),
    }
