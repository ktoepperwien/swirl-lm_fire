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
"""A library for the turbulent kinetic energy (TKE) modeling (JAX port).

JAX port of `swirl_lm.physics.combustion.turbulent_kinetic_energy`.
"""


import jax.numpy as jnp
from swirl_lm.jax.utility import types
from swirl_lm.physics.combustion import turbulent_kinetic_energy_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


def _update_local_halos(value: ScalarField, halo_width: int) -> ScalarField:
  """Pads halos with symmetric condition to `value`."""
  if halo_width < 1:
    raise ValueError(
        f'`halo_width` has to be greater than 1. {halo_width} is provided.'
    )
  interior = value[
      halo_width:-halo_width,
      halo_width:-halo_width,
      halo_width:-halo_width,
  ]
  return jnp.pad(
      interior,
      pad_width=[(halo_width, halo_width)] * 3,
      mode='symmetric',
  )


def _local_box_filter_3d(state: ScalarField) -> ScalarField:
  """Applies a uniform box filter of width 3 to `state` locally.

  This computes the average of the 3x3x3 stencil centered at each point,
  assuming halos of the input are valid.

  Args:
    state: A 3D array to be filtered.

  Returns:
    The filtered 3D array.
  """
  # Uniform 3x3x3 average via cumulative sum trick along each axis.
  result = state
  for axis in range(3):
    # Shift and average along this axis.
    shifted_m = jnp.roll(result, 1, axis=axis)
    shifted_p = jnp.roll(result, -1, axis=axis)
    result = (shifted_m + result + shifted_p) / 3.0
  return result


def constant_tke_update_function(tke_value: float):
  """Generates an update function for TKE with a constant value.

  Args:
    tke_value: A constant that specifies the value of the TKE.

  Returns:
    A function that updates TKE to the constant value.
  """

  def additional_states_update_fn(
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarFieldMap:
    """Updates 'tke' in `additional_states`."""
    del states
    tke = tke_value * jnp.ones_like(additional_states['tke'])
    updated: dict[str, ScalarField] = {}
    for key, value in additional_states.items():
      updated[key] = tke if key == 'tke' else value
    return updated

  return additional_states_update_fn


def algebraic_tke_update_function():
  """Generates a function that updates TKE algebraically.

  With this model, the TKE is computed with velocity fluctuations estimated by
  the filtered quantities following a scale similarity approximation in
  turbulence.

  Returns:
    A function that updates TKE with the current velocity field.
  """

  def additional_states_update_fn(
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarFieldMap:
    """Updates 'tke' in `additional_states`."""
    u_mean = _local_box_filter_3d(states['u'])
    v_mean = _local_box_filter_3d(states['v'])
    w_mean = _local_box_filter_3d(states['w'])

    tke = 0.5 * (
        (states['u'] - u_mean) ** 2
        + (states['v'] - v_mean) ** 2
        + (states['w'] - w_mean) ** 2
    )

    updated: dict[str, ScalarField] = {}
    for key, value in additional_states.items():
      updated[key] = _local_box_filter_3d(tke) if key == 'tke' else value
    return updated

  return additional_states_update_fn


def turbulent_viscosity_tke_update_function():
  """Generates a function that updates TKE from turbulent viscosity.

  Reference: Pressel et al. 2015, J. Advances in Modeling Earth Systems.

  Note that `nu_t` and `tke` are required in the `additional_states`.

  Returns:
    A function that updates TKE with the current velocity field.
  """
  c_k = 0.1

  def additional_states_update_fn(
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      dx: float,
      dy: float,
      dz: float,
      halo_width: int,
  ) -> ScalarFieldMap:
    """Updates 'tke' in `additional_states`."""
    del states
    if 'nu_t' not in additional_states:
      raise ValueError(
          '`nu_t` is required to use the turbulence viscosity to compute TKE.'
      )

    delta = (dx * dy * dz) ** (1.0 / 3.0)
    nu_t = additional_states['nu_t']
    tke = (nu_t / (c_k * delta)) ** 2

    updated: dict[str, ScalarField] = {}
    for key, value in additional_states.items():
      updated[key] = (
          _update_local_halos(tke, halo_width) if key == 'tke' else value
      )
    return updated

  return additional_states_update_fn


def tke_update_fn_manager(
    tke_update_option: turbulent_kinetic_energy_pb2.TKE,
):
  """Generates the TKE update function requested by `tke_update_option`.

  Args:
    tke_update_option: The method that is used to update the TKE.

  Returns:
    The update function for TKE.
  """
  oneof = tke_update_option.WhichOneof('tke_model_option')
  if oneof == 'constant':
    return constant_tke_update_function(tke_update_option.constant.tke_constant)
  elif oneof == 'algebraic':
    return algebraic_tke_update_function()
  elif oneof == 'turbulent_viscosity':
    return turbulent_viscosity_tke_update_function()
  else:
    raise ValueError(
        f'Undefined TKE model {tke_update_option}. Available models are: '
        '"constant", "algebraic", "turbulent_viscosity".'
    )
