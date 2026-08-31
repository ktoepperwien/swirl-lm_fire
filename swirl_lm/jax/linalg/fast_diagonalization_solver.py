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
r"""A JAX direct solver for a Kronecker sum of three Hermitian matrices.

Uses the tensor product method to solve a linear system
(A₁ ⊗ I ⊗ I + I ⊗ A₂ ⊗ I + I ⊗ I ⊗ A₃) x = b directly using
the tensor product eigendecomposition [1],

  (A₁ ⊗ I ⊗ I + I ⊗ A₂ ⊗ I + I ⊗ I ⊗ A₃)⁻¹
  = (V₁ ⊗ V₂ ⊗ V₃) (Λ₁ ⊗ I ⊗ I + I ⊗ Λ₂ ⊗ I + I ⊗ I ⊗ Λ₃)⁻¹ (V₁ ⊗ V₂ ⊗ V₃)⁻¹,

where V₁ and Λ₁ are the eigenvectors and eigenvalues of A₁:
  A₁ V₁ = V₁ Λ₁.

[1] Lynch, Robert E., John R. Rice, and Donald H. Thomas. "Direct solution of
partial difference equations by tensor product methods." Numerische Mathematik
6.1 (1964): 185-199.
"""



from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.linalg import base_poisson_solver
from swirl_lm.jax.linalg import poisson_solver_pb2
from swirl_lm.jax.utility import get_kernel_fn
from swirl_lm.jax.utility import grid_parametrization
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
PoissonSolverSolution = base_poisson_solver.PoissonSolverSolution

_N_DIM = 3
_DTYPE = jnp.float32
_CTYPE = jnp.complex64


def _make_laplacian_matrix(
    n: int,
    dx: float,
    bc_low: str,
    bc_high: str,
) -> np.ndarray:
  """Constructs the 1D Laplacian matrix with boundary conditions.

  The Laplacian is discretized as: (u_{i-1} - 2*u_i + u_{i+1}) / dx^2.

  Boundary treatments:
  - NEUMANN: The homogeneous Neumann BC implies p_{ghost} = p_{boundary},
    so the Laplacian at the boundary becomes (-p_0 + p_1) / dx^2 (low) or
    (-p_{n-1} + p_{n-2}) / dx^2 (high). The diagonal entry becomes -1.
  - DIRICHLET: p_{ghost} = -p_{boundary} (antisymmetric mirror). The
    standard [-1, 2, -1] stencil already handles this.
  - PERIODIC: The matrix wraps around: a[0, -1] = 1 and a[-1, 0] = 1.

  Args:
    n: The number of interior grid points.
    dx: The grid spacing.
    bc_low: Boundary condition at the lower end ('DIRICHLET', 'NEUMANN', or
      'PERIODIC').
    bc_high: Boundary condition at the higher end ('DIRICHLET', 'NEUMANN', or
      'PERIODIC').

  Returns:
    An n*n numpy array representing the 1D Laplacian operator.
  """
  a = np.zeros((n, n), dtype=np.float64)

  # Interior: standard second-order finite difference [-1, 2, -1] / dx^2.
  for i in range(n):
    a[i, i] = -2.0
    if i > 0:
      a[i, i - 1] = 1.0
    if i < n - 1:
      a[i, i + 1] = 1.0

  # Handle PERIODIC BCs: wrap the matrix around.
  if bc_low == 'PERIODIC' or bc_high == 'PERIODIC':
    if bc_low != bc_high:
      raise ValueError(
          'Periodic boundary condition is ambiguous for Laplacian matrix '
          f'construction: ({bc_low}, {bc_high}). Both ends must be PERIODIC.'
      )
    a[0, -1] = 1.0
    a[-1, 0] = 1.0
  else:
    # Handle NEUMANN BCs: p_{ghost} = p_{boundary}.
    # This changes the diagonal from -2 to -1.
    if bc_low == 'NEUMANN':
      a[0, 0] = -1.0

    if bc_high == 'NEUMANN':
      a[-1, -1] = -1.0

    # DIRICHLET: no modification needed (default stencil is correct).

  a /= dx * dx
  return a


def fast_diagonalization_solve_fn(
    a: Sequence[np.ndarray],
    cutoff: float,
) -> Callable[[jax.Array], jax.Array]:
  """Creates a direct solver for a Hermitian Kronecker sum.

  This version works on local (single-partition) data. For distributed
  execution, the caller is responsible for gathering/scattering data.

  Args:
    a: The linear operators stored in a length 3 list, with each element to be
      applied in dimension 0, 1, and 2, respectively.
    cutoff: The threshold for the absolute eigenvalue to be considered as 0.

  Returns:
    A callable `solve(rhs) -> solution` that solves the linear system.
  """
  if len(a) != _N_DIM:
    raise ValueError(f'Expected {_N_DIM} linear operators, but got {len(a)}.')

  # Eigendecomposition of each 1D operator (done once at init time, in numpy).
  lam = []
  v = []
  for i in range(_N_DIM):
    eigenvalues, eigenvectors = np.linalg.eigh(a[i])
    lam.append(eigenvalues)
    v.append(eigenvectors)

  # Precompute the sum of eigenvalues: Λ₁ ⊗ I ⊗ I + I ⊗ Λ₂ ⊗ I + I ⊗ I ⊗ Λ₃
  lam_sum = (
      lam[0].reshape(-1, 1, 1)
      + lam[1].reshape(1, -1, 1)
      + lam[2].reshape(1, 1, -1)
  )

  # Pseudoinverse of the eigenvalue sum (zero out near-zero eigenvalues).
  lam_inv = np.where(
      np.abs(lam_sum) > cutoff,
      1.0 / np.where(np.abs(lam_sum) > cutoff, lam_sum, 1.0),
      0.0,
  )

  # Convert to JAX arrays.
  lam_inv_jax = jnp.array(lam_inv, dtype=_DTYPE)
  # Eigenvector matrices.
  v_jax = [jnp.array(v[i], dtype=_DTYPE) for i in range(_N_DIM)]
  vt_jax = [jnp.array(v[i].T, dtype=_DTYPE) for i in range(_N_DIM)]

  def solve(rhs: jax.Array) -> jax.Array:
    """Solves the Kronecker-sum linear system.

    Args:
      rhs: The right-hand-side 3D array.

    Returns:
      The solution 3D array.
    """
    # Step 1: Forward transform: buf = (V₁ᵀ ⊗ V₂ᵀ ⊗ V₃ᵀ) rhs
    # Apply V₁ᵀ along dim 0: buf[i,j,k] = Σ_n Vᵀ₁[i,n] * rhs[n,j,k]
    buf = jnp.einsum('in,njk->ijk', vt_jax[0], rhs)
    # Apply V₂ᵀ along dim 1: buf[i,j,k] = Σ_n Vᵀ₂[j,n] * buf[i,n,k]
    buf = jnp.einsum('jn,ink->ijk', vt_jax[1], buf)
    # Apply V₃ᵀ along dim 2: buf[i,j,k] = Σ_n Vᵀ₃[k,n] * buf[i,j,n]
    buf = jnp.einsum('kn,ijn->ijk', vt_jax[2], buf)

    # Step 2: Divide by eigenvalues.
    buf = lam_inv_jax * buf

    # Step 3: Inverse transform: res = (V₁ ⊗ V₂ ⊗ V₃) buf
    # Apply V₁ along dim 0: res[i,j,k] = Σ_n V₁[i,n] * buf[n,j,k]
    res = jnp.einsum('in,njk->ijk', v_jax[0], buf)
    # Apply V₂ along dim 1: res[i,j,k] = Σ_n V₂[j,n] * res[i,n,k]
    res = jnp.einsum('jn,ink->ijk', v_jax[1], res)
    # Apply V₃ along dim 2: res[i,j,k] = Σ_n V₃[k,n] * res[i,j,n]
    res = jnp.einsum('kn,ijn->ijk', v_jax[2], res)

    return res

  return solve


class FastDiagonalizationSolver(base_poisson_solver.PoissonSolver):
  """A Poisson solver using the Fast Diagonalization (direct) method."""

  def __init__(
      self,
      grid_params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      solver_option: poisson_solver_pb2.PoissonSolver,
  ):
    """Initializes the fast diagonalization solver.

    Args:
      grid_params: The grid parametrization.
      kernel_op: An object holding a library of kernel operations.
      solver_option: The option of the selected solver (PoissonSolver proto).
    """
    super().__init__(grid_params, kernel_op, solver_option)
    self._fd_option = solver_option.fast_diagonalization
    self._cutoff = self._fd_option.cutoff
    self._halo_width = self._fd_option.halo_width or grid_params.halo_width

    # Determine boundary conditions from proto.
    bc_low = self._fd_option.boundary_condition_low
    bc_high = self._fd_option.boundary_condition_high

    def _bc_name(bc_type: int) -> str:
      """Converts proto BoundaryConditionType to a string."""
      # BC_TYPE_DIRICHLET=1, BC_TYPE_NEUMANN=2, BC_TYPE_PERIODIC=4,
      # BC_TYPE_NEUMANN_2=5
      if bc_type == 1:
        return 'DIRICHLET'
      elif bc_type == 4:
        return 'PERIODIC'
      # Default: NEUMANN (covers BC_TYPE_NEUMANN=2, BC_TYPE_NEUMANN_2=5,
      # and unset/unknown=0).
      return 'NEUMANN'

    # Build Laplacian matrices for each dimension.
    assert grid_params.nx is not None
    assert grid_params.ny is not None
    assert grid_params.nz is not None
    assert grid_params.dx is not None
    assert grid_params.dy is not None
    assert grid_params.dz is not None

    nx = grid_params.nx - 2 * self._halo_width
    ny = grid_params.ny - 2 * self._halo_width
    nz = grid_params.nz - 2 * self._halo_width

    a_matrices = [
        _make_laplacian_matrix(
            nx,
            float(grid_params.dx),
            _bc_name(bc_low.dim_x if bc_low else 0),
            _bc_name(bc_high.dim_x if bc_high else 0),
        ),
        _make_laplacian_matrix(
            ny,
            float(grid_params.dy),
            _bc_name(bc_low.dim_y if bc_low else 0),
            _bc_name(bc_high.dim_y if bc_high else 0),
        ),
        _make_laplacian_matrix(
            nz,
            float(grid_params.dz),
            _bc_name(bc_low.dim_z if bc_low else 0),
            _bc_name(bc_high.dim_z if bc_high else 0),
        ),
    ]

    self._solve_fn = fast_diagonalization_solve_fn(a_matrices, self._cutoff)

  def solve(
      self,
      rhs: ScalarField,
      p0: ScalarField,
      mesh: jax.sharding.Mesh,
      halo_update_fn=None,
      additional_states=None,
  ) -> PoissonSolverSolution:
    """Solves the Poisson equation using fast diagonalization.

    Args:
      rhs: A 3D array that represents the right hand side tensor.
      p0: A 3D array that provides initial guess (unused by direct solver).
      mesh: A jax Mesh object representing the device topology.
      halo_update_fn: A function that updates the halo of the input.
      additional_states: Additional static fields needed in the computation.

    Returns:
      A dict with solution and placeholder values for iterations/norms.
    """
    hw = self._halo_width

    # Strip halos from rhs.
    rhs_interior = rhs[hw:-hw, hw:-hw, hw:-hw]

    # Solve the interior system.
    solution_interior = self._solve_fn(rhs_interior)

    # Pad the solution back with zeros in the halo region.
    solution = jnp.pad(
        solution_interior,
        pad_width=((hw, hw), (hw, hw), (hw, hw)),
        mode='constant',
        constant_values=0.0,
    )

    # Update halos if a halo update function is provided.
    if halo_update_fn is not None:
      solution = halo_update_fn(solution)

    return {
        base_poisson_solver.X: solution,
        base_poisson_solver.RESIDUAL_L2_NORM: jnp.array(-1.0),
        base_poisson_solver.COMPONENT_WISE_DISTANCE: jnp.array(-1.0),
        base_poisson_solver.ITERATIONS: jnp.array(1),
    }

  def residual(
      self,
      p: ScalarField,
      rhs: ScalarField,
      mesh: jax.sharding.Mesh,
      additional_states=None,
  ) -> jax.Array:
    """Computes the residual (LHS - RHS) of the Poisson equation.

    Args:
      p: The approximate solution.
      rhs: The right hand side.
      mesh: A jax Mesh object.
      additional_states: Additional fields.

    Returns:
      The residual (Laplacian(p) - rhs).
    """
    halo_update = base_poisson_solver._halo_update_homogeneous_neumann(  # pylint: disable=protected-access
        mesh, self._grid_params
    )
    p_updated = halo_update(p)
    return self._laplacian(p_updated) - rhs
