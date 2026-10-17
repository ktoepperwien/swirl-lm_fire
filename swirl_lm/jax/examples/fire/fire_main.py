from absl import app
from absl import flags
from absl import logging
from etils import epath
from google.protobuf import text_format
import jax

from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.core import driver as driver_lib
from swirl_lm.jax.examples.fire import fire

jax.config.update("jax_enable_x64", False)

_NUM_CYCLES = flags.DEFINE_integer(
    'num_cycles',
    1,
    'number of cycles to run. Each cycle generates a set of output',
)
_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 1, 'number of steps to run before generating an output.'
)
_CONFIG_FILEPATH = flags.DEFINE_string(
    'config_filepath', None,
    'The full path to a config file.',
    allow_override=True
)
_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir', None,
    'The full path to a checkpoint directory.',
    allow_override=True
)
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'checkpoint_interval',
    10,
    'interval at which to write solution.',
)
_RESTART_FROM = flags.DEFINE_string(
    'restart_from',
    None,
    'A zarr checkpoint to resume from, such as '
    '<checkpoint_dir>/step_000500.zarr. The requested steps are run on top of '
    'the step that checkpoint records, so the step index continues and the '
    'resumed run does not overwrite the output it started from. Writing into '
    'the same --checkpoint_dir is therefore fine.',
)
_RESTART_STEP = flags.DEFINE_integer(
    'restart_step',
    None,
    'Overrides the step index to continue from. Needed only to renumber '
    'deliberately, or to resume a checkpoint written before the step was '
    'recorded in its metadata.',
)

FLAGS = flags.FLAGS


def _make_params() -> parameters_lib.SwirlLMParameters:
  """Builds the fire configuration from the config file."""
  # Initialize empty instance
  config = parameters_pb2.SwirlLMParameters()

  with open(FLAGS.config_filepath, "r") as file:
    text_data = file.read()
    text_format.Parse(text_data, config)
  return parameters_lib.SwirlLMParameters(config)


def _check_restart_path() -> str | None:
  """Returns the checkpoint to resume from, failing early if it is missing.

  A mistyped path would otherwise surface as an obscure zarr error after the
  job has queued, compiled and placed the state.

  Returns:
    The checkpoint path, or None when this is not a restart.

  Raises:
    ValueError: If the path does not exist.
  """
  path = _RESTART_FROM.value
  if path is None:
    return None
  if not epath.Path(path).exists():
    raise ValueError(f'--restart_from={path} does not exist.')
  return path


def main(_):
  params = _make_params()
  restart_from = _check_restart_path()

  simulation = fire.Fire(params)
  params.source_update_fn_lib = simulation.source_term_update_fn()
  params.additional_states_update_fn = simulation.additional_states_update_fn
  params.preprocessing_states_update_fn = simulation.pre_simulation_update_fn

  if restart_from is not None:
    logging.info(
        'Resuming from %s; %d step(s) will be run on top of its step.',
        restart_from,
        FLAGS.num_cycles * FLAGS.num_steps,
    )

  # Run the simulation using the JAX driver.
  driver_lib.run_simulation(
      params=params,
      init_fn=simulation.initialization,
      num_steps=FLAGS.num_cycles * FLAGS.num_steps,
      num_steps_per_cycle=FLAGS.num_steps,
      checkpoint_dir=FLAGS.checkpoint_dir,
      checkpoint_interval=FLAGS.checkpoint_interval,
      restart_from=restart_from,
      restart_step=_RESTART_STEP.value,
  )


if __name__ == '__main__':
  app.run(main)
