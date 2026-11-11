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
"""Joins a multi-process JAX cluster, so a run can span more than one node.

JAX is multi-controller: every node runs this same program, and
`jax.distributed.initialize` is how those peers rendezvous and agree on one
global device list rather than each process seeing only its own GPUs which
would run two independent simulations.

Encapsulated here because it is a process-global, once-only side effect that
has to precede every call touching the XLA backend (`jax.devices` and
`jax.device_count` included). Call `initialize()` as the first statement of
`main()`.

The read-side counterpart is `global_layout.to_host_array`, which gathers a
state array that spans processes.
"""

import os

from absl import logging
import jax

# Launcher variables carrying the process count. JAX detects the launchers
# themselves. If any of these variables is set, `jax.distributed.initialize()`
# needs to run. Else the environment is outside a cluster and/or a single-
# process run.
_PROCESS_COUNT_VARS = (
    'SLURM_NTASKS',
    'SLURM_NPROCS',
    'OMPI_COMM_WORLD_SIZE',
    'PMI_SIZE',
)

# Forces multi-process on ('1') or off ('0'), bypassing detection. A hung
# start is ambiguous between the rendezvous and the job itself, and setting
# this to 0 separates the two.
_OVERRIDE_VAR = 'SWIRL_LM_MULTIPROCESS'


def launcher_process_count() -> int:
  """Returns the process count this job was launched with, 1 if unlaunched.

  Read from the environment rather than from `jax.process_count`, which is
  only correct after `initialize` and which initializes the backend.
  """
  for var in _PROCESS_COUNT_VARS:
    value = os.environ.get(var)
    if value:
      try:
        return int(value)
      except ValueError:
        logging.warning('%s is set to %r, which is not a count.', var, value)
  return 1


def initialize() -> None:
  """Joins the cluster, if this process was launched as part of one.

  Idempotent, and a no-op outside a multi-process launch, so entry points can
  call it unconditionally.
  """
  if jax.distributed.is_initialized():
    return

  override = os.environ.get(_OVERRIDE_VAR)
  if override == '0':
    logging.info('%s=0, staying single-process.', _OVERRIDE_VAR)
    return
  if override != '1' and launcher_process_count() <= 1:
    return

  # No arguments: the coordinator address, the process count and this
  # process's id are all derived from the launcher's environment. On Slurm the
  # address is the first host of `SLURM_STEP_NODELIST` with a port taken from
  # the job id, and `local_device_ids` becomes `[SLURM_LOCALID]`, which is why
  # one task per GPU needs no `CUDA_VISIBLE_DEVICES`.
  jax.distributed.initialize()

  logging.info(
      'Joined cluster as process %d of %d, holding %d of %d %s devices.',
      jax.process_index(),
      jax.process_count(),
      jax.local_device_count(),
      jax.device_count(),
      jax.default_backend(),
  )
