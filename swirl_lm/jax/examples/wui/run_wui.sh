#!/bin/bash
#
# Runs the wildland-urban-interface case.
#
#   SCENE=/path/to/wui_scene.npz ./run_wui.sh      # one device
#   SCENE=... CORES_X=4 ./run_wui.sh               # four devices
#   JAX_PLATFORMS=cuda SCENE=... CORES_X=4 ./run_wui.sh
#   WIND=10 SCENE=... ./run_wui.sh                 # a fire-weather wind
#   PYTHON=/path/to/python ./run_wui.sh
#
# The mesh comes from the scene file, not from the case file; re-run
# `extract_scene.py` to change it.
#
# The decomposition is CORES_X x CORES_Y x 1, and it is the only thing that
# decides how many devices are used, on CPU and on GPU alike. There is
# deliberately no separate "number of devices" knob: `driver.create_mesh`
# takes `jax.devices()[:cores_x*cores_y*cores_z]`, so a device count that
# disagrees with the decomposition either fails or, worse, silently leaves
# devices idle. On CPU this script derives the simulated device count from
# the decomposition; on GPU the scheduler decides, and `wui_main` warns if
# the two disagree.
#
# CORES_Z is not offered. The terrain is a height field, which reports the
# vertical gap rather than the true distance, so a ghost cell's interpolation
# stencil reaches about five cells along z -- further than any reasonable
# halo. Horizontally it reaches under three, which `halo_width: 4` covers.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

CONFIG="${CONFIG:-${HERE}/wui.pbtxt}"
OUTPUT="${OUTPUT:-${HERE}/SOLUT}"
NUM_CYCLES="${NUM_CYCLES:-40}"
NUM_STEPS="${NUM_STEPS:-500}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
PYTHON="${PYTHON:-python}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# Resolve the decomposition the same way the run will: the case file, with
# any environment override applied. Reading it here rather than guessing is
# what keeps the simulated CPU device count in step with it.
read -r CX CY CZ NCORES < <(
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON}" - "${CONFIG}" <<'PY'
import sys, os
from google.protobuf import text_format
from swirl_lm.jax.examples.wui import wui_pb2

case = wui_pb2.WuiCase()
with open(sys.argv[1]) as f:
  text_format.Parse(f.read(), case)
cx = int(os.environ.get('CORES_X') or case.cores_x)
cy = int(os.environ.get('CORES_Y') or case.cores_y)
cz = int(case.cores_z)
print(cx, cy, cz, cx * cy * cz)
PY
)

if [[ "${JAX_PLATFORMS}" == *"cuda"* || "${JAX_PLATFORMS}" == *"gpu"* ]]; then
  # cuDNN has no algorithm for the stencil convolutions this solver emits, so
  # autotuning fails outright on GPU. See MEMORY.md, kernel-operator defects.
  export XLA_FLAGS="--xla_gpu_autotune_level=0 ${XLA_FLAGS:-}"
else
  export XLA_FLAGS="--xla_force_host_platform_device_count=${NCORES} ${XLA_FLAGS:-}"
fi

ARGS=(
  --config_filepath="${CONFIG}"
  --checkpoint_dir="${OUTPUT}"
  --checkpoint_interval="${CHECKPOINT_INTERVAL}"
  --num_cycles="${NUM_CYCLES}"
  --num_steps="${NUM_STEPS}"
  --cores_x="${CX}"
  --cores_y="${CY}"
  --cores_z="${CZ}"
)
[[ -n "${SCENE:-}" ]] && ARGS+=(--scene_path="${SCENE}")
[[ -n "${WIND:-}" ]] && ARGS+=(--inflow_speed="${WIND}")

mkdir -p "${OUTPUT}"
cd "${ROOT}"
echo "backend ${JAX_PLATFORMS}, decomposition ${CX}x${CY}x${CZ} = ${NCORES} device(s)"
echo "running: ${PYTHON} -m swirl_lm.jax.examples.wui.wui_main ${ARGS[*]}"
PYTHONPATH="${ROOT}:${PYTHONPATH:-}" exec "${PYTHON}" -m \
  swirl_lm.jax.examples.wui.wui_main "${ARGS[@]}"
