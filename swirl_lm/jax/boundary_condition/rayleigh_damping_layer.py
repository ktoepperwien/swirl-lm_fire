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
"""A library of the Rayleigh damping (sponge) layer (JAX port).

This is the JAX port of `swirl_lm.boundary_condition.rayleigh_damping_layer`.

This boundary treatment is applied as a forcing term that takes the form [1]:
  f(φ) = -β (φ - φ₀),
where β is the coefficient that determines where the sponge layer is applied,
φ₀ is the desired value at the sponge layer.

To use this library, one or more variables for 'beta' coefficients are required
in the `additional_states`.

Reference:
1. Durran, Dale R., and Joseph B. Klemp. 1983. "A Compressible Model for the
   Simulation of Moist Mountain Waves." Monthly Weather Review 111 (12):
   2341-61.
"""


from collections.abc import Callable, Iterable, Sequence
from typing import Optional, Union

from absl import logging
import jax.numpy as jnp
import numpy as np
from swirl_lm.boundary_condition import rayleigh_damping_layer_pb2
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

_Orientation = rayleigh_damping_layer_pb2.RayleighDampingLayer.Orientation
_SpongeInfo = rayleigh_damping_layer_pb2.RayleighDampingLayer.VariableInfo
_RayleighDampingLayerSeq = Sequence[
    rayleigh_damping_layer_pb2.RayleighDampingLayer
]
# Function (xx, yy, zz, lx, ly, lz) -> ScalarField.
InitFn = Callable[
    [ScalarField, ScalarField, ScalarField, float, float, float],
    ScalarField,
]


def get_sponge_force_name(varname: str) -> str:
  """Generates the name of the forcing term for the input variable name.

  Args:
    varname: The name of the variable for which the name of the forcing term is
      requested.

  Returns:
    The name of the forcing term, which takes the form 'src_[varname]'.
  """
  return f'src_{varname}'


def get_sponge_target_name(varname: str) -> str:
  """Generates the name of the target value term for the input variable name.

  Args:
    varname: The name of the variable for which the name of the target value is
      requested.

  Returns:
    The name of the target term, which takes the form 'sponge_target_[varname]'.
  """
  return f'sponge_target_{varname}'


def variable_type_lib_from_proto(
    sponges: _RayleighDampingLayerSeq,
) -> dict[str, bool]:
  """Generates a library for the type of the variable from the proto.

  Args:
    sponges: A sequence of materialized sponge layer protos.

  Returns:
    A dictionary with keys being the variable names, and values being the type
    of the variable (primitive or conservative).
  """
  out = {}
  for sponge in sponges:
    for info in sponge.variable_info:
      out[info.name] = info.primitive
  return out


def _get_beta_name_from_sponge_info(
    sponge: rayleigh_damping_layer_pb2.RayleighDampingLayer,
) -> str:
  # Use default name 'sponge_beta' if beta_name is not explicitly set.
  return sponge.beta_name or 'sponge_beta'


def beta_name_by_var(
    sponges: _RayleighDampingLayerSeq,
) -> dict[str, str]:
  """Returns the mapping from variable names to beta variable names."""
  out = {}
  seen_beta_names: set[str] = set()
  for sponge in sponges:
    beta_name = _get_beta_name_from_sponge_info(sponge)
    if beta_name in seen_beta_names:
      raise ValueError(
          f'Sponge beta variable `{beta_name}` is defined more than once.'
      )
    seen_beta_names.add(beta_name)
    for info in sponge.variable_info:
      if info.name in out:
        raise ValueError(
            f'Variable `{info.name}` participates in multiple sponge layers.'
        )
      out[info.name] = beta_name
  return out


def sponge_info_map(
    sponges: _RayleighDampingLayerSeq,
) -> dict[str, _SpongeInfo]:
  """Returns a mapping from variable names to their sponge info."""
  out = {}
  for sponge in sponges:
    for info in sponge.variable_info:
      out[info.name] = info
  return out


def target_value_mean_dims_by_var(
    sponges: _RayleighDampingLayerSeq,
    periodic_dims: Optional[Sequence[bool]],
) -> dict[str, Sequence[int]]:
  """Returns the mapping from variable names to target value mean dimensions."""
  out: dict[str, Sequence[int]] = {}
  for sponge in sponges:
    target_value_mean_dims = list(sponge.target_value_mean_dim)
    if not target_value_mean_dims and periodic_dims is not None:
      target_value_mean_dims = [i for i, val in enumerate(periodic_dims) if val]
    for info in sponge.variable_info:
      out[info.name] = target_value_mean_dims
  return out


def klemp_lilly_relaxation_coeff_fn(
    orientation: Iterable[_Orientation],
    x0: float,
    y0: float,
    z0: float,
) -> InitFn:
  """Generates a function that computes 'sponge_beta' (Klemp & Lilly, 1978).

  The sponge layer coefficient is defined as:
  beta = 0, if h <= h_d
  beta = a_max sin^2 (pi/2 (h - h_d) / (h_t - h_d)), if h > h_d.

  Args:
    orientation: A sequence of orientation elements, each with dim, fraction,
      a_coeff, and face fields.
    x0: Coordinate of face 0 along the x dimension.
    y0: Coordinate of face 0 along the y dimension.
    z0: Coordinate of face 0 along the z dimension.

  Returns:
    The `sponge_beta` coefficient following Klemp & Lilly, 1978.

  Raises:
    ValueError: If `dim` in `orientation` is not one of 0, 1, or 2.
    ValueError: If the sponge layer fraction is below 0 or above 1.
  """

  def init_fn(
      xx: ScalarField,
      yy: ScalarField,
      zz: ScalarField,
      lx: float,
      ly: float,
      lz: float,
  ) -> ScalarField:
    """Initializes the sponge layer relaxation coefficient beta."""
    beta = jnp.zeros_like(xx)

    for sponge in orientation:
      dim = sponge.dim

      if dim == 0:
        grid = xx
        h_t = lx
        c0 = x0
      elif dim == 1:
        grid = yy
        h_t = ly
        c0 = y0
      elif dim == 2:
        grid = zz
        h_t = lz
        c0 = z0
      else:
        raise ValueError(
            f'Dimension has to be one of 0, 1, and 2. {dim} is given.'
        )

      if sponge.fraction < 0 or sponge.fraction > 1:
        raise ValueError(
            'The fraction of sponge layer should be in (0, 1). '
            f'{sponge.fraction} is given in dim {dim}.'
        )

      a_max = np.reciprocal(sponge.a_coeff)

      # Set the default face to the higher end for backward compatibility.
      face = sponge.face if sponge.HasField('face') else 1

      if face == 1:
        h_d = (1.0 - sponge.fraction) * h_t + c0
        buf = jnp.where(
            grid <= h_d,
            jnp.zeros_like(grid),
            a_max
            * jnp.power(jnp.sin(np.pi / 2.0 * (grid - h_d) / (h_t - h_d)), 2),
        )
      elif face == 0:
        h_d = sponge.fraction * h_t + c0
        buf = jnp.where(
            grid >= h_d,
            jnp.zeros_like(grid),
            a_max
            * jnp.power(jnp.sin(np.pi / 2.0 * jnp.abs(grid - h_d) / h_d), 2),
        )
      else:
        raise ValueError(
            f'Face index has to be one of 0 and 1. {face} is provided.'
        )

      beta = jnp.maximum(beta, buf)

    return beta

  return init_fn


def klemp_lilly_relaxation_coeff_fns_for_sponges(
    sponge_infos: _RayleighDampingLayerSeq,
    x0: float,
    y0: float,
    z0: float,
) -> dict[str, InitFn]:
  """Returns a mapping from beta name to its initialization function."""
  return {
      _get_beta_name_from_sponge_info(
          sponge_info
      ): klemp_lilly_relaxation_coeff_fn(sponge_info.orientation, x0, y0, z0)
      for sponge_info in sponge_infos
  }


class RayleighDampingLayer:
  """A library of the sponge layer (JAX port)."""

  def __init__(
      self,
      sponge_infos: _RayleighDampingLayerSeq,
      periodic_dims: Optional[Sequence[bool]] = None,
  ):
    """Initializes the sponge layer library.

    Args:
      sponge_infos: Sequence of materialized RayleighDampingLayer protos.
      periodic_dims: An optional list of booleans indicating the periodic
        dimensions.
    """
    self._is_primitive = variable_type_lib_from_proto(sponge_infos)
    self._beta_name_by_var = beta_name_by_var(sponge_infos)
    self._sponge_info_map = sponge_info_map(sponge_infos)

    self._target_value_mean_dims_by_var = target_value_mean_dims_by_var(
        sponge_infos, periodic_dims
    )

    logging.info(
        'Sponge layer will be applied for the following variables with '
        'following values: %s',
        self._sponge_info_map,
    )

  def _get_sponge_force(
      self,
      field: ScalarField,
      beta: ScalarField,
      dt: float,
      target_value_mean_dims: Sequence[int],
      target_state: Optional[Union[float, ScalarField]] = None,
  ) -> ScalarField:
    """Computes the sponge forcing term.

    Args:
      field: The value of the variable to which the sponge forcing is applied.
      beta: The coefficients to be applied as the sponge.
      dt: The time step size.
      target_value_mean_dims: Dimensions over which to compute the target value
        mean.
      target_state: An optional reference state from which to compute the sponge
        target.

    Returns:
      The sponge force.
    """
    if target_state is not None:
      target_value = target_state
    else:
      # Single-device: mean over specified dims replaces global_mean.
      if target_value_mean_dims:
        target_value = jnp.mean(
            field, axis=tuple(target_value_mean_dims), keepdims=True
        )
      else:
        target_value = jnp.mean(field)

    return beta / dt * (target_value - field)

  @property
  def varnames(self) -> tuple[str, ...]:
    """Generates a tuple of variable names to which sponge is applied."""
    return tuple(self._sponge_info_map.keys())

  def generate_initial_states(
      self,
      grid_shape: tuple[int, int, int],
      beta_fields: dict[str, ScalarField],
  ) -> dict[str, ScalarField]:
    """Generates the required initial fields by the simulation.

    Args:
      grid_shape: The shape of the 3D computational grid.
      beta_fields: A mapping from sponge beta variable names to their
        precomputed coefficient fields.

    Returns:
      A dictionary of state variables that are required by the Rayleigh damping
      layer.
    """
    output: dict[str, ScalarField] = dict(beta_fields)
    for variable in self._sponge_info_map:
      output[get_sponge_force_name(variable)] = jnp.zeros(grid_shape)
    return output

  def additional_states_update_fn(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      dt: float,
  ) -> dict[str, ScalarField]:
    """Updates the forcing term due to the sponge layer.

    The forcing term will be added to the existing forcing term in
    `additional_states` for variables that are in the scope of
    `self._sponge_info_map`.

    Args:
      states: A keyed dictionary of states that will be updated.
      additional_states: A dictionary of states needed by the update fn.
      dt: The time step size.

    Returns:
      A dictionary with updated sponge-layer forcing.

    Raises:
      ValueError: if beta variable or `target_state_name` is not present in
      `additional_states`.
    """
    beta_names_not_in_additional_states = set(
        self._beta_name_by_var.values()
    ) - set(additional_states.keys())
    if beta_names_not_in_additional_states:
      raise ValueError(
          f'{beta_names_not_in_additional_states} not found '
          'in `additional_states.`'
      )

    additional_states_updated: dict[str, ScalarField] = dict(additional_states)
    for varname, var_info in self._sponge_info_map.items():
      if varname not in states:
        logging.warning(
            '%s is not a valid state. Available states are: %r',
            varname,
            list(states.keys()),
        )
        continue

      sponge_name = get_sponge_force_name(varname)
      target_val: Optional[Union[float, ScalarField]] = None
      if var_info.HasField('target_state_name'):
        if var_info.target_state_name not in additional_states_updated:
          raise ValueError(
              f'Target_state_name {var_info.target_state_name} is not among '
              'the states.'
          )
        target_val = additional_states_updated[var_info.target_state_name]
      elif var_info.HasField('target_value'):
        target_val = var_info.target_value

      sponge_force = self._get_sponge_force(
          states[varname],
          additional_states[self._beta_name_by_var[varname]],
          dt,
          self._target_value_mean_dims_by_var[varname],
          target_val,
      )
      if not self._is_primitive[varname]:
        sponge_force = states['rho'] * sponge_force
      additional_states_updated[sponge_name] = (
          additional_states_updated[sponge_name] + sponge_force
      )

    return additional_states_updated
