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

# Copyright 2024 Google LLC
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
"""Updates flow field variables and source terms by combustion (JAX port).

JAX port of `swirl_lm.physics.combustion.combustion`.
"""


import jax
import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.physics.combustion import biofuel_multistep
from swirl_lm.jax.physics.combustion import turbulent_kinetic_energy
from swirl_lm.jax.physics.combustion import wood
from swirl_lm.jax.utility import types
from swirl_lm.physics.combustion import turbulent_kinetic_energy_pb2

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap


def _compute_tke(
    states: ScalarFieldMap,
    additional_states: ScalarFieldMap,
    tke_params: turbulent_kinetic_energy_pb2.TKE,
) -> ScalarField:
  """Computes the turbulent kinetic energy."""
  tke_update_fn = turbulent_kinetic_energy.tke_update_fn_manager(tke_params)

  additional_states_tke = dict(additional_states)
  additional_states_tke['tke'] = additional_states.get(
      'tke', jnp.zeros_like(states['u'])
  )
  additional_states_tke = tke_update_fn(states, additional_states_tke)
  return additional_states_tke['tke']


def combustion_step(
    replica_id: jax.Array,
    replicas: np.ndarray,
    step_id: int,
    states: ScalarFieldMap,
    additional_states: ScalarFieldMap,
    params: parameters_lib.SwirlLMParameters,
) -> ScalarFieldMap:
  """Updates states and computes source terms due to combustion.

  Args:
    replica_id: The id of the replica.
    replicas: A numpy array that maps grid coordinates to replica id numbers.
    step_id: The index of the current time step.
    states: A keyed dictionary of states that will be updated.
    additional_states: Additional states needed by the update fn.
    params: An instance of the Swirl-LM simulation global context.

  Returns:
    A dictionary with updated flow field variables and/or source terms due to
    combustion.

  Raises:
    AssertionError: If not all required additional states are provided.
    ValueError: If TKE model is not defined in the config for wood combustion.
  """
  del step_id
  if params.combustion is None:
    return {}

  additional_states_combustion: dict[str, jax.Array] = {}

  if params.combustion.HasField('wood'):
    model = wood.wood_combustion_factory(params)

    required_additional_states = list(
        model.required_additional_states_keys(states)
    )
    # Remove 'tke' since it will be provided by this function.
    required_additional_states.remove('tke')
    assert set(required_additional_states).issubset(additional_states.keys()), (
        'Required additional states missing for the wood combustion model.'
        f' Required: {required_additional_states}. Missing:'
        f' {set(required_additional_states) - set(additional_states.keys())}.'
    )

    if (
        params.combustion is not None
        and params.combustion.HasField('wood')
        and params.combustion.wood.HasField('tke')
    ):
      tke = _compute_tke(
          states,
          additional_states,
          params.combustion.wood.tke,
      )
    else:
      raise ValueError(
          'A TKE model is required for wood combustion, but is undefined in the'
          ' config.'
      )

    additional_states_combustion.update({
        key: val
        for key, val in additional_states.items()
        if key in required_additional_states
    })
    additional_states_combustion['tke'] = tke
    if 'p_ref' in additional_states:
      additional_states_combustion['p_ref'] = additional_states['p_ref']
    if 'theta_ref' in additional_states:
      additional_states_combustion['theta_ref'] = additional_states['theta_ref']
    additional_states_combustion.update(
        model.update_fn(additional_states.get('rho_f_init'))(
            replica_id,
            replicas,
            states,
            additional_states_combustion,
        )
    )

  if params.combustion.HasField('biofuel_multistep'):
    bm_model = biofuel_multistep.BiofuelMultistep(params)

    required_additional_states = list(
        bm_model.required_additional_states_keys(states)
    )
    required_additional_states.remove('tke')
    assert set(required_additional_states).issubset(additional_states.keys()), (
        'Required additional states missing for the biofuel combustion model.'
        f' Required: {required_additional_states}. Missing:'
        f' {set(required_additional_states) - set(additional_states.keys())}.'
    )

    combustion_model = (
        params.combustion.biofuel_multistep.pyrolysis_char_oxidation.wood
    )
    if combustion_model.HasField('tke'):
      tke = _compute_tke(
          states,
          additional_states,
          combustion_model.tke,
      )
    else:
      raise ValueError(
          'A TKE model is required for wood combustion, but is undefined in the'
          ' config.'
      )

    additional_states_combustion.update({
        key: val
        for key, val in additional_states.items()
        if key in required_additional_states
    })
    additional_states_combustion['tke'] = tke
    if 'rho_f_init' in additional_states:
      additional_states_combustion['rho_f_init'] = additional_states[
          'rho_f_init'
      ]
    if 'p_ref' in additional_states:
      additional_states_combustion['p_ref'] = additional_states['p_ref']
    if 'theta_ref' in additional_states:
      additional_states_combustion['theta_ref'] = additional_states['theta_ref']
    additional_states_combustion.update(
        bm_model.additional_states_update_fn(
            replica_id,
            replicas,
            states,
            additional_states_combustion,
        )
    )
  return additional_states_combustion
