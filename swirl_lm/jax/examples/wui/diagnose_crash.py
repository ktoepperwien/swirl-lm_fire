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
"""Says where a run went non-finite, and what the geometry is like there.

`driver.run_simulation` saves a checkpoint before it raises on non-finite
velocity, so the evidence is already on disk. Point this at that checkpoint
and the scene it came from:

    python -m swirl_lm.jax.examples.wui.diagnose_crash \
        --checkpoint=SOLUT/step_000500.zarr --scene=wui_scene.npz

It reports whether the damage is one cell or everywhere, where the worst
cells sit relative to the ground and the nearest building, and how steep
the terrain is under them. A handful of cells on a steep slope beside a
building is a local geometry problem; a whole plane is a boundary problem.
"""

from absl import app
from absl import flags
import numpy as np
import zarr

from swirl_lm.jax.examples.wui import wui
from swirl_lm.jax.examples.wui import wui_pb2

_CHECKPOINT = flags.DEFINE_string('checkpoint', None, 'Crash checkpoint.')
_SCENE = flags.DEFINE_string('scene', None, 'The scene file it was run on.')
_TOP = flags.DEFINE_integer('top', 12, 'How many worst cells to list.')


def main(_):
  sim = wui.Wui(wui_pb2.WuiCase(scene_path=_SCENE.value))
  store = zarr.open(_CHECKPOINT.value, mode='r')
  u, v, w = (np.asarray(store[k][:]) for k in ('u', 'v', 'w'))
  fluid = np.asarray(store['ib_interior_mask'][:]) > 0.5

  speed = np.sqrt(u ** 2 + v ** 2 + w ** 2)
  bad = ~np.isfinite(speed)
  print('non-finite cells: %d of %d (%.4f%%); in the fluid: %d'
        % (bad.sum(), bad.size, 100.0 * bad.mean(), (bad & fluid).sum()))
  if bad.any():
    for axis, name in enumerate('xyz'):
      extent = np.unique(np.argwhere(bad)[:, axis])
      print('  spans %d of %d planes along %s (%d..%d)'
            % (len(extent), bad.shape[axis], name, extent.min(),
               extent.max()))

  # Where the damage starts: the largest finite speeds are the cells that
  # were on their way out.
  finite = np.where(np.isfinite(speed) & fluid, speed, 0.0)
  terrain = sim.terrain_height
  gx, gy = np.gradient(terrain, sim.dx, sim.dy)
  slope = np.degrees(np.arctan(np.hypot(gx, gy)))
  reach = np.hypot(sim.box_half_extents[:, 0], sim.box_half_extents[:, 1])

  print('\nfluid speed: mean %.2f  p99 %.2f  max finite %.2f'
        % (finite[finite > 0].mean(), np.percentile(finite[finite > 0], 99),
           finite.max()))
  print('\n  speed      x     y     z   terrain  above_gnd  slope  bldg_gap')
  for flat in np.argsort(finite.ravel())[::-1][:_TOP.value]:
    i, j, k = np.unravel_index(flat, speed.shape)
    x, y, z = i * sim.dx, j * sim.dy, k * sim.dz
    gap = (np.hypot(sim.box_centres[:, 0] - x,
                    sim.box_centres[:, 1] - y) - reach).min()
    print('  %7.2f %5.0f %5.0f %5.0f %8.1f %9.1f %6.1f %9.1f'
          % (finite[i, j, k], x, y, z, terrain[i, j],
             z - terrain[i, j], slope[i, j], gap))
  print('\nA negative building gap means the cell is inside a building\'s '
        'footprint in plan view.')


if __name__ == '__main__':
  flags.mark_flags_as_required(['checkpoint', 'scene'])
  app.run(main)
