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
"""Runs the wildland-urban-interface case.

The config file holds a `WuiCase`, not a `SwirlLMParameters`; the solver
configuration is derived from it and from the scene file so that the mesh and
the geometry defined on it cannot disagree. `--dump_config` writes the derived
configuration out so a run records exactly what it solved.
"""

import os

from absl import app
from absl import flags
from absl import logging
from google.protobuf import text_format
import jax

from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.examples.wui import wui
from swirl_lm.jax.examples.wui import wui_pb2

jax.config.update('jax_enable_x64', False)

_NUM_CYCLES = flags.DEFINE_integer(
    'num_cycles',
    1,
    'number of cycles to run. Each cycle generates a set of output',
)
_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 1, 'number of steps to run before generating an output.'
)
_CONFIG_FILEPATH = flags.DEFINE_string(
    'config_filepath',
    None,
    'The full path to a WuiCase config file.',
    allow_override=True,
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    None,
    'The full path to a checkpoint directory.',
    allow_override=True,
)
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'checkpoint_interval',
    10,
    'interval at which to write solution.',
)
_SCENE_PATH = flags.DEFINE_string(
    'scene_path',
    None,
    'Overrides the scene file in the config. This is the `.npz` written by '
    '`extract_scene.py`, and it defines the mesh as well as the geometry.',
)
_INFLOW_SPEED = flags.DEFINE_float(
    'inflow_speed', None, 'Overrides the wind speed in the config.'
)
_CORES_X = flags.DEFINE_integer(
    'cores_x', None, 'Overrides cores_x in the config.'
)
_CORES_Y = flags.DEFINE_integer(
    'cores_y', None, 'Overrides cores_y in the config.'
)
_CORES_Z = flags.DEFINE_integer(
    'cores_z', None, 'Overrides cores_z in the config.'
)
_DUMP_CONFIG = flags.DEFINE_bool(
    'dump_config',
    True,
    'Write the derived solver configuration next to the checkpoints.',
)
_LOG_POISSON_RESIDUAL = flags.DEFINE_bool(
    'log_poisson_residual',
    False,
    'Report what the pressure solve achieved, once per sub-iteration: the '
    'relative residual `||A dp - b|| / ||b||`, and what the correction did to '
    'the continuity residual. Off by default: the switch is read while '
    'tracing, so unset, neither the extra operator application nor the host '
    'callback that prints it enters the compiled graph at all.',
)

FLAGS = flags.FLAGS


def _make_case() -> wui_pb2.WuiCase:
  """Reads the case file, applying any command-line overrides."""
  case = wui_pb2.WuiCase()
  if FLAGS.config_filepath:
    with open(FLAGS.config_filepath, 'r') as f:
      text_format.Parse(f.read(), case)
  if _SCENE_PATH.value is not None:
    case.scene_path = _SCENE_PATH.value
  if _INFLOW_SPEED.value is not None:
    case.inflow_speed = _INFLOW_SPEED.value
  for flag, field in (
      (_CORES_X, 'cores_x'), (_CORES_Y, 'cores_y'), (_CORES_Z, 'cores_z')
  ):
    if flag.value is not None:
      setattr(case, field, flag.value)
  return case


def _check_device_count(case: wui_pb2.WuiCase) -> None:
  """Compares the requested decomposition with the devices actually present.

  `driver.create_mesh` takes `jax.devices()[:cx*cy*cz]`, so asking for fewer
  cores than there are devices silently leaves the rest idle -- a four-GPU job
  quietly running on one.

  Args:
    case: The resolved case parameters.

  Raises:
    ValueError: If there are fewer devices than the decomposition needs.
  """
  wanted = case.cores_x * case.cores_y * case.cores_z
  available = jax.device_count()
  if available < wanted:
    raise ValueError(
        f'The configuration asks for {wanted} cores '
        f'({case.cores_x}x{case.cores_y}x{case.cores_z}) but JAX sees '
        f'{available} device(s) on the {jax.default_backend()} backend. '
        'On GPU, request more devices from the scheduler; on CPU, set '
        f'XLA_FLAGS=--xla_force_host_platform_device_count={wanted}.'
    )
  if available > wanted:
    logging.warning(
        'Using %d of %d available %s device(s): the configuration asks for '
        '%dx%dx%d. The remaining %d will sit idle. Set --cores_x/--cores_y/'
        '--cores_z so their product is %d to use them all.',
        wanted, available, jax.default_backend(),
        case.cores_x, case.cores_y, case.cores_z,
        available - wanted, available,
    )


def main(_):
  case = _make_case()
  _check_device_count(case)
  simulation = wui.Wui(case)
  params = simulation.params
  params.log_poisson_residual = _LOG_POISSON_RESIDUAL.value
  params.source_update_fn_lib = simulation.source_term_update_fn()
  logging.info('%s', simulation.summary())

  if _DUMP_CONFIG.value and FLAGS.checkpoint_dir:
    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
    path = os.path.join(FLAGS.checkpoint_dir, 'derived_config.pbtxt')
    with open(path, 'w') as f:
      f.write(simulation.config_text())
    logging.info('Derived solver configuration written to %s', path)

  mesh = driver_lib.create_mesh(params)
  geometry = simulation.ghost_points(mesh)
  fallback = simulation.fallback_fraction(geometry)
  if fallback > 0.01:
    logging.warning(
        '%.1f%% of the ghost cells are on the nearest-fluid fallback, which '
        'is first-order where the rest are second-order. That usually means '
        'a surface lying on the cell centres or terrain too steep for the '
        'vertical spacing.',
        100.0 * fallback,
    )

  driver_lib.run_simulation(
      params=params,
      init_fn=simulation.initialization,
      num_steps=_NUM_CYCLES.value * _NUM_STEPS.value,
      num_steps_per_cycle=_NUM_STEPS.value,
      mesh=mesh,
      checkpoint_dir=FLAGS.checkpoint_dir,
      checkpoint_interval=_CHECKPOINT_INTERVAL.value,
  )
  logging.info(
      'done: %d steps, %.1f s of simulated time',
      _NUM_CYCLES.value * _NUM_STEPS.value,
      _NUM_CYCLES.value * _NUM_STEPS.value * simulation.dt,
  )


if __name__ == '__main__':
  app.run(main)
