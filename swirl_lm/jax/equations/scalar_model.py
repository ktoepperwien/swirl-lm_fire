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

"""Scalar model factory."""

from __future__ import annotations

from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.equations import scalar_model_generic
from swirl_lm.jax.equations import scalar_model_humidity
from swirl_lm.jax.equations import scalar_model_potential_temperature
from swirl_lm.jax.equations import scalar_model_total_energy
from swirl_lm.jax.physics.turbulence import sgs_model as sgs_model_lib
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap
ScalarModel = scalar_model_generic.ScalarModel


def scalar_model_factory(
    params: parameters_lib.SwirlLMParameters,
    scalar_name: str,
    sgs: sgs_model_lib.SgsModel | None = None,
) -> scalar_model_generic.ScalarModel:
  """Creates the appropriate scalar model for the given scalar name.

  Dispatches to specialised models for known scalar types:
    - 'theta', 'theta_li' -> PotentialTemperature
    - humidity variables   -> Humidity
    - 'e_t'               -> TotalEnergy
    - all others           -> GenericScalarModel

  Args:
    params: Simulation parameters.
    scalar_name: Name of the scalar variable.
    sgs: Optional SGS model.

  Returns:
    A ScalarModel instance for the given scalar.
  """
  if (
      scalar_name
      in scalar_model_potential_temperature.POTENTIAL_TEMPERATURE_VARNAMES
  ):
    return scalar_model_potential_temperature.PotentialTemperature(
        params, scalar_name, sgs
    )
  elif scalar_name in scalar_model_humidity.HUMIDITY_VARNAMES:
    return scalar_model_humidity.Humidity(params, scalar_name, sgs)
  elif scalar_name == scalar_model_total_energy.TOTAL_ENERGY_VARNAME:
    return scalar_model_total_energy.TotalEnergy(params, scalar_name, sgs)
  else:
    return scalar_model_generic.GenericScalarModel(params, scalar_name, sgs)
