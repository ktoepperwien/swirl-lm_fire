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
"""Wind through a wildland-urban interface.

The scene is a real neighbourhood -- 24 houses along a road through a valley,
with vegetation -- reduced to three layers by `extract_scene.py`, each matched
to what the source data can actually support:

  * **buildings** as a union of oriented boxes, whose signed distance is
    exact. The input is a Gaussian-splat classification, so every surface is a
    shell with no interior and nothing may depend on a watertight geometry;
  * **terrain** as a height field, one smooth single-valued surface;
  * **vegetation** as a drag field, because foliage is porous and a solid
    canopy would block the flow it is supposed to slow down.

Usage mirrors the cylinder example: the config file holds a `WuiCase` and the
solver configuration is derived from it.

  ```python
  simulation = wui.Wui(case)
  params = simulation.params
  params.source_update_fn_lib = simulation.source_term_update_fn()
  state = driver.run_simulation(
      params=params,
      init_fn=simulation.initialization,
      num_steps=2000,
  )
  ```

**This is not a validation case and no number from it should be quoted.**
`MEMORY.md` section 5: the pressure solve does not converge. Only Jacobi runs
at all, and on the cylinder the Strouhal number kept moving as the iteration
count rose. The pictures will look right; the magnitudes are not to be
trusted. `pressure_iterations` is therefore the first knob to question if
anything looks wrong.

Two properties of the scene drive the configuration, both measured:

**`cores_z` must be 1.** A height field returns the vertical gap rather than
the true distance, so on terrain whose median slope is 19 degrees the image
point of a ghost cell can sit seven cells away along z, past any reasonable
halo. The x and y reach is under two cells, so those are the axes to split.

**The vertical spacing must stay at 1 m.** At 2 x 2 x 1 m the ghost-point
precompute reports no shortened probes and no fallback cells on this terrain;
at 2 x 2 x 2 m a third of the ghost cells fall back to a nearest-fluid value.
"""

import json
import math
from typing import Any, Callable

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

from google.protobuf import text_format
from swirl_lm.base import parameters_pb2
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.boundary_condition import immersed_boundary_ghost_point
from swirl_lm.jax.boundary_condition import rayleigh_damping_layer
from swirl_lm.jax.examples.wui import wui_pb2
from swirl_lm.jax.utility import types


ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

VELOCITY = ('u', 'v', 'w')

# `convection.py` and `interpolation.py` build their own kernel operator with
# this width regardless of the configuration, so every per-core extent has to
# be divisible by it as well as by the configured `kernel_size`.
_CONVECTION_KERNEL_SIZE = 4

# Fuel densities below this are treated as no vegetation at all, matching the
# gate in the fire example.
_EPSILON = 1e-6

# How far a surface must stay from a cell centre, in cells. A surface passing
# exactly through a cell centre is the degenerate case of the ghost-point
# construction: the body intercept coincides with the ghost cell, the image
# point lands on it too, the interpolation weight on the cell itself goes to
# one, and the implicit relation `1 - zeta * w_self` is singular. Those cells
# drop to the nearest-fluid fallback. It is not a rare accident here -- the
# FastFuels elevation is whole metres and the vertical spacing is 1 m, so
# before this was handled 18% of the terrain's ghost cells fell back.
_NODE_CLEARANCE_CELLS = 0.05
# Every box is grown by this fraction of a cell, for the same reason: it
# moves the faces off the cell centres they would otherwise share with the
# grid. At 1 m this is 13 mm on a house, four orders of magnitude below the
# fit error the boxes already carry.
_FACE_NUDGE_CELLS = 0.013


class Wui:
  """Wind through a neighbourhood, with a ghost-point immersed boundary."""

  def __init__(self, case: wui_pb2.WuiCase | None = None):
    """Loads the scene and derives the solver configuration from it.

    Args:
      case: The case parameters. Defaults are used for anything unset.

    Raises:
      ValueError: If the case is not self-consistent, or if the scene file
        does not describe a grid that the requested decomposition can carry.
    """
    self.case = case or wui_pb2.WuiCase()
    if not self.case.scene_path:
      raise ValueError(
          '`scene_path` is required: it is the `.npz` written by '
          '`extract_scene.py`, and it defines the mesh as well as the '
          'geometry.'
      )
    self._validate()

    scene = np.load(self.case.scene_path, allow_pickle=True)
    self.box_centres = np.asarray(scene['box_centres'], np.float64)
    self.box_half_extents = np.asarray(scene['box_half_extents'], np.float64)
    self.box_rotations = np.asarray(scene['box_rotations_deg'], np.float64)
    self.terrain_height = np.asarray(scene['terrain_height'], np.float32)
    self.fuel = np.asarray(scene['fuel_bulk_density'], np.float32)
    self.origin_utm = np.asarray(scene['origin_utm'], np.float64)
    self.spacing = np.asarray(scene['spacing'], np.float64)
    self.metadata = json.loads(str(scene['metadata']))

    self.fx, self.fy, self.fz = (int(n) for n in scene['shape'])
    self.dx, self.dy, self.dz = (float(d) for d in self.spacing)

    # The solver's nodes span `[0, length]` inclusive, so its spacing is
    # `length / (n - 1)` and not `length / n`. Declaring the scene's extent
    # as the length would therefore stretch every cell by `n / (n - 1)` --
    # 0.6% here, which would quietly make `dz` 1.005 m when the measured
    # statement about this terrain is specifically about 1 m. Taking a node
    # per scene cell and one fewer interval instead keeps the spacing exact
    # and leaves a cell-wide sliver of the scene unused at the far edges.
    self.lx = (self.fx - 1) * self.dx
    self.ly = (self.fy - 1) * self.dy
    self.lz = (self.fz - 1) * self.dz
    self._check_scene()

    self.u_ref = self.case.inflow_speed
    # The fastest wind in the domain, not the reference speed: a log law keeps
    # growing with height, and over a domain this deep the inlet reaches
    # roughly twice its value at 10 m. The time step follows that and the
    # smallest spacing, which is the vertical metre -- the same combination
    # the driver's CFL diagnostic reports, so the number it prints is
    # comparable with the `cfl` asked for here. It is still only a starting
    # estimate: the flow accelerates over the ridge and between the houses,
    # so the reported CFL comes out above this.
    self.u_max = float(self.inflow_profile(self.lz))
    self.dt = (
        self.case.dt
        if self.case.HasField('dt')
        else self.case.cfl * min(self.dx, self.dy, self.dz) / self.u_max
    )

    self.params = self._build_params()
    self.gp = self.params.grid_params

    # Has to follow the grid, which decides where the cell centres are, and
    # precede the geometry, which is built on the shifted ground.
    self.terrain_shift = self._terrain_offset()

    self.sponge = (
        rayleigh_damping_layer.RayleighDampingLayer(self.params.sponge)
        if self.params.sponge
        else None
    )

    # The geometry comes from the generated configuration, so the solver and
    # anything derived here describe the same scene.
    self.geometry_fn = immersed_boundary_ghost_point.geometry_from_proto(
        self.params.boundary_models.ib.ghost_point.geometry,
        height_fields={'terrain': self.terrain_profile()},
    )
    self._ghost_points: Any = None

  # -- Validation -----------------------------------------------------------

  def _validate(self) -> None:
    """Checks the case parameters before anything is built from them."""
    case = self.case
    if case.inflow_speed <= 0.0:
      raise ValueError(
          f'`inflow_speed` must be positive, got {case.inflow_speed}.'
      )
    if case.roughness_length <= 0.0:
      raise ValueError(
          '`roughness_length` must be positive: it appears inside a '
          f'logarithm, got {case.roughness_length}.'
      )
    if case.reference_height <= case.roughness_length:
      raise ValueError(
          f'`reference_height` ({case.reference_height} m) must exceed '
          f'`roughness_length` ({case.roughness_length} m); the log-law '
          'profile is not defined below the roughness height.'
      )
    if case.halo_width < 2:
      raise ValueError(
          f'`halo_width` is {case.halo_width}; the QUICK convection scheme '
          'needs 2, and the ghost-point stencil may need more.'
      )
    if case.cores_z != 1:
      raise ValueError(
          f'`cores_z` is {case.cores_z}, but the terrain height field makes '
          'the vertical the one axis that must not be split: it reports the '
          'vertical gap rather than the true distance, so on this terrain a '
          'ghost cell\'s image point can sit 7 cells away along z, far '
          f'beyond a halo of {case.halo_width}. Split x or y instead, where '
          'the reach is under 2 cells.'
      )

  def _check_scene(self) -> None:
    """Checks the scene file against itself and against the vertical spacing.

    Raises:
      ValueError: If the arrays in the scene disagree, or if the vertical
        spacing is one the terrain cannot be trusted at.
    """
    shapes = {
        'box_centres': self.box_centres.shape,
        'box_half_extents': self.box_half_extents.shape,
        'box_rotations_deg': self.box_rotations.shape,
    }
    if len(set(shapes.values())) != 1:
      raise ValueError(f'The box arrays disagree in shape: {shapes}.')
    if self.terrain_height.shape != (self.fx, self.fy):
      raise ValueError(
          f'`terrain_height` is {self.terrain_height.shape} but the scene '
          f'declares a {self.fx} x {self.fy} horizontal mesh.'
      )
    if self.fuel.shape != (self.fx, self.fy, self.fz):
      raise ValueError(
          f'`fuel_bulk_density` is {self.fuel.shape} but the scene declares '
          f'{self.fx} x {self.fy} x {self.fz}.'
      )
    if self.box_half_extents.size and self.box_half_extents.min() <= 0.0:
      raise ValueError('Every box half extent must be positive.')

    # No box may reach a lateral domain face. A building that touches one has
    # the face's own boundary condition and the immersed boundary's no-slip
    # contending for the same cells, with nothing to arbitrate, and the run
    # goes non-finite in under ten steps. Measured on an 80 m crop of this
    # scene: with six of fourteen boxes crossing a face it diverged at step
    # 10, and dropping exactly those six made it stable at ten steps. This
    # is why `extract_scene.py` pads the mesh with clear fluid.
    if self.box_centres.size:
      # A box rotated about z reaches `hypot(hx, hy)` from its centre.
      reach = np.hypot(
          self.box_half_extents[:, 0], self.box_half_extents[:, 1]
      )
      room = np.minimum(
          np.minimum(self.box_centres[:, 0] - reach,
                     self.lx - self.box_centres[:, 0] - reach),
          np.minimum(self.box_centres[:, 1] - reach,
                     self.ly - self.box_centres[:, 1] - reach),
      )
      clearance = float(room.min())
      if clearance < self.dx:
        raise ValueError(
            f'{int((room < self.dx).sum())} of {len(room)} building boxes '
            f'come within {self.dx:g} m of a lateral domain face (closest '
            f'{clearance:.1f} m). The face boundary condition and the '
            'immersed boundary then contend for the same cells and the run '
            'diverges within ten steps. Re-run `extract_scene.py` with a '
            'larger `--margin`.'
        )

    # A height field degrades once the terrain slope exceeds `dz/dx`, and this
    # scene is steep: median slope 19 degrees, 90th percentile 35. Measured,
    # 1 m is clean and 2 m is not, so refuse to run coarser than the spacing
    # the scene was extracted at rather than discover it from a third of the
    # ghost cells falling back.
    if self.dz > 1.0:
      raise ValueError(
          f'The scene has dz = {self.dz} m. On this terrain anything above '
          '1 m puts a large fraction of the ghost cells onto a '
          'nearest-fluid fallback (measured: 35% at 2 m). Re-run '
          '`extract_scene.py` with `--dz=1`.'
      )

  def _grid_size(self, interior: int, cores: int, axis: str) -> int:
    """Per-core extent including halos, with the divisibility checks.

    Args:
      interior: Interior points along this axis, over the whole domain.
      cores: Devices along this axis.
      axis: Axis name, for the error messages.

    Returns:
      The per-core extent including halos.

    Raises:
      ValueError: If the domain does not split evenly, or if the resulting
        extent is not divisible by a kernel width in use.
    """
    if interior % cores:
      raise ValueError(
          f'The {axis} direction has {interior} interior points, which does '
          f'not divide evenly across {cores} cores. The mesh is fixed by the '
          'scene file, so change the decomposition or re-run '
          '`extract_scene.py` over a different crop.'
      )
    extent = interior // cores + 2 * self.case.halo_width

    # A per-core extent that is not divisible by a kernel width sends that
    # kernel down a slice fallback which disagrees with the convolution path
    # for some stencils. Nothing warns. See MEMORY.md, kernel-operator
    # defects. `extract_scene.py` rounds every axis to a multiple of eight so
    # that this holds for the usual decompositions, but it cannot know the
    # decomposition, so check here.
    for width, source in (
        (self.case.kernel_size, '`kernel_size`'),
        (_CONVECTION_KERNEL_SIZE, 'the width convection.py hard-codes'),
    ):
      if extent % width:
        raise ValueError(
            f'The {axis} direction has {extent} points per core, which is '
            f'not divisible by {width} ({source}). That silently selects a '
            'finite-difference path that disagrees with the usual one for '
            f'some stencils. Use a different number of cores along {axis} '
            f'({interior} interior points), or change `kernel_size`.'
        )
    return extent

  # -- Geometry -------------------------------------------------------------

  def _terrain_offset(self) -> float:
    """A vertical shift that keeps the terrain off the grid's cell centres.

    Chosen by measurement rather than picked: the shift that maximises the
    clearance between the terrain surface and the nearest cell centre,
    searched over one cell. Returns 0 when the terrain already clears the
    centres, which is the usual case for a floating-point elevation model
    and emphatically not the case for this one.

    Returns:
      The shift in metres, in `[0, dz)`.
    """
    # The surface only matters where the grid samples it, so evaluate it at
    # the node columns rather than over the table.
    x = np.asarray(self.gp.global_xyz_with_halos[0])
    y = np.asarray(self.gp.global_xyz_with_halos[1])
    profile = self._raw_terrain_profile()
    xx, yy = np.meshgrid(x, y, indexing='ij')
    height = np.asarray(profile(jnp.asarray(xx), jnp.asarray(yy)), np.float64)

    candidates = np.arange(0.0, 1.0, 0.005) * self.dz
    # Distance from each shifted surface height to the nearest cell centre,
    # in cells; `0.5 - |frac - 0.5|` is that distance on a unit lattice.
    frac = ((height[None, :, :] + candidates[:, None, None]) / self.dz) % 1.0
    clearance = np.minimum(frac, 1.0 - frac).reshape(len(candidates), -1)
    worst = clearance.min(axis=1)
    best = int(np.argmax(worst))
    if worst[0] >= _NODE_CLEARANCE_CELLS:
      return 0.0
    if worst[best] < _NODE_CLEARANCE_CELLS:
      logging.warning(
          'No vertical shift keeps the terrain clear of the cell centres: '
          'the best of %d candidates leaves %.4f cells, under the %.2f '
          'wanted. Expect ghost cells on the nearest-fluid fallback where '
          'the surface crosses a cell centre.',
          len(candidates), worst[best], _NODE_CLEARANCE_CELLS,
      )
    logging.info(
        'terrain raised by %.3f m so it clears every cell centre by %.3f '
        'cells (it cleared %.3f unshifted; the elevation model is whole '
        'metres and the grid is %.3g m)',
        candidates[best], worst[best], worst[0], self.dz,
    )
    return float(candidates[best])

  def _raw_terrain_profile(self) -> Callable[[Any, Any], Any]:
    """The terrain as the scene file gives it, with no node-clearance shift.

    Returns:
      A function of `(x, y)` returning the terrain height.
    """
    """The terrain elevation `h(x, y)`, bilinearly interpolated.

    Takes positions in the solver frame -- metres from the south-west corner
    of the domain floor -- and returns metres above that floor, which is the
    frame `extract_scene.py` writes everything in. Works on scalars, which is
    how `signed_distance.height_field` calls it, and on arrays, which is how
    the initial condition does; and it is differentiable, which is what
    `signed_distance.normal` needs.

    Outside the table it holds the edge value, so a halo cell beyond the
    domain sees flat ground rather than an extrapolated cliff.

    Returns:
      A function of `(x, y)` returning the terrain height.
    """
    table = jnp.asarray(self.terrain_height)
    nx, ny = self.terrain_height.shape
    dx, dy = self.dx, self.dy

    def h(x, y):
      # Sample `i` of the table is the cell centred at `(i + 0.5) * dx`.
      fx = jnp.clip(x / dx - 0.5, 0.0, nx - 1.0)
      fy = jnp.clip(y / dy - 0.5, 0.0, ny - 1.0)
      i0 = jnp.floor(fx).astype(jnp.int32)
      j0 = jnp.floor(fy).astype(jnp.int32)
      i1 = jnp.minimum(i0 + 1, nx - 1)
      j1 = jnp.minimum(j0 + 1, ny - 1)
      tx, ty = fx - i0, fy - j0
      return (
          (1.0 - tx) * (1.0 - ty) * table[i0, j0]
          + tx * (1.0 - ty) * table[i1, j0]
          + (1.0 - tx) * ty * table[i0, j1]
          + tx * ty * table[i1, j1]
      )

    return h

  def terrain_profile(self) -> Callable[[Any, Any], Any]:
    """The terrain elevation `h(x, y)` used by everything in this case.

    Takes positions in the solver frame -- metres from the south-west corner
    of the domain floor -- and returns metres above that floor, which is the
    frame `extract_scene.py` writes everything in. Works on scalars, which is
    how `signed_distance.height_field` calls it, and on arrays, which is how
    the initial condition does; and it is differentiable, which is what
    `signed_distance.normal` needs.

    Includes the node-clearance shift, so the geometry and the initial
    condition see the same ground.

    Returns:
      A function of `(x, y)` returning the terrain height.
    """
    profile = self._raw_terrain_profile()
    shift = self.terrain_shift
    if not shift:
      return profile
    return lambda x, y: profile(x, y) + shift

  def inflow_profile(self, height: Any) -> Any:
    """The log-law wind speed at a height above the local ground.

    Args:
      height: Height above the terrain, in metres. Negative heights -- cells
        inside the ground -- give zero.

    Returns:
      The streamwise speed.
    """
    z0 = self.case.roughness_length
    h = jnp.maximum(height, 0.0)
    scale = math.log((self.case.reference_height + z0) / z0)
    return self.u_ref * jnp.log((h + z0) / z0) / scale

  # -- Configuration --------------------------------------------------------

  def _geometry_text(self) -> str:
    """The scene as `IBGeometry` messages: one box per part, plus the terrain.

    Written out in full rather than passed around in Python so that the
    dumped configuration is a complete record of what was solved. Bodies of
    one type are batched by `geometry_from_proto`, so the cost of listing
    seventy boxes is seventy table rows, not seventy traced callables.

    Returns:
      The `geometry` blocks, indented to sit inside `ghost_point`.
    """
    # Grown off the cell centres; see `_FACE_NUDGE_CELLS`.
    nudge = _FACE_NUDGE_CELLS * min(self.dx, self.dy, self.dz)
    blocks = []
    for centre, half, rotation in zip(
        self.box_centres, self.box_half_extents + nudge, self.box_rotations
    ):
      blocks.append(f"""\
      geometry {{
        box {{
          center {{ x: {centre[0]:.4g} y: {centre[1]:.4g} z: {centre[2]:.4g} }}
          half_extents {{ x: {half[0]:.4g} y: {half[1]:.4g} z: {half[2]:.4g} }}
          rotation_deg {{ z: {rotation[2]:.4g} }}
        }}
      }}""")
    # `profile` is a name the example resolves, which is how an arbitrary
    # elevation function reaches a geometry described in a proto.
    blocks.append('      geometry { height_field { profile: "terrain" } }')
    return '\n'.join(blocks)

  def _sponge_config(self) -> str:
    """The Rayleigh damping block, indented to sit inside `boundary_models`.

    One sponge at the outlet, and optionally one on each lateral face. Each
    relaxes the velocity towards its initial value -- the undisturbed inlet
    profile -- over the outer `sponge_fraction` of that axis.

    The outlet one is what stops a recirculation bubble reaching the
    outflow plane, where a zero-gradient condition cannot represent it. The
    lateral ones stand in for periodicity, which this terrain rules out:
    the south and north edges differ by 16 m on average and 47 m at worst.

    Returns:
      The `sponge` block, or an empty string when damping is switched off.
    """
    case = self.case
    if case.sponge_fraction <= 0.0:
      return ''
    faces = [(0, 1, case.sponge_fraction)]        # the outlet
    if case.lateral_sponge_fraction > 0.0:
      faces += [(1, 0, case.lateral_sponge_fraction),
                (1, 1, case.lateral_sponge_fraction)]
    orientations = '\n'.join(
        f'    orientation {{ dim: {dim} face: {face} '
        f'fraction: {fraction:g} a_coeff: {case.sponge_coeff:g} }}'
        for dim, face, fraction in faces
    )
    variables = '\n'.join(
        f'    variable_info {{ name: "{name}" '
        f'target_state_name: "{name}_init" primitive: true }}'
        for name in VELOCITY
    )
    return f"""  sponge {{
{orientations}
{variables}
  }}"""

  def _solver_text(self) -> str:
    """The Poisson solver block, indented to sit inside `solver {}`.

    `pressure_iterations` is the iteration count for all three, so it means
    sweeps, iterations or V-cycles depending on which is selected and the
    useful magnitudes differ accordingly.

    Returns:
      The configuration text for the selected solver.

    Raises:
      ValueError: If the solver is unrecognised.
    """
    case = self.case
    solver = case.solver
    if solver == wui_pb2.WuiCase.JACOBI:
      return (
          f'    jacobi {{\n'
          f'      max_iterations: {case.pressure_iterations}\n'
          f'      halo_width: {case.halo_width}\n'
          f'      omega: 0.67\n'
          f'    }}'
      )
    if solver == wui_pb2.WuiCase.CONJUGATE_GRADIENT:
      # `atol` is relative, which `l2_norm_reduction` is what selects: the
      # norm of the right hand side scales with the mesh and the flow, so one
      # absolute number means a converged solve on this grid and an immediate
      # stop on another. `reprojection` is left unset deliberately --
      # `Pressure` decides it from the boundary conditions once they are
      # settled, and with an outlet pinning the level it must be off.
      return (
          f'    conjugate_gradient {{\n'
          f'      max_iterations: {case.pressure_iterations}\n'
          f'      halo_width: {case.halo_width}\n'
          f'      atol: 1e-6\n'
          f'      l2_norm_reduction: true\n'
          f'    }}'
      )
    if solver == wui_pb2.WuiCase.MULTIGRID:
      # Two weighted Jacobi sweeps either side of the coarse correction, the
      # damping of 2/3 being the standard smoother choice, and a coarsest
      # grid of eight: the Galerkin coarse operator spans five points per
      # axis and four cells cannot hold it. These are the values the fire
      # case was measured at; none of them is exposed because nothing has yet
      # given a reason to vary them.
      return (
          f'    multigrid {{\n'
          f'      num_iterations: {case.pressure_iterations}\n'
          f'      n_smooth: 2\n'
          f'      weight: 0.6667\n'
          f'      n_coarse: 1\n'
          # `dim_0/1/2`, not `dim_x/y/z`: this text is parsed against the
          # TensorFlow `swirl_lm.CoordinateInt`, whose fields are numbered.
          # `simulation._COORDINATE_ALIASES` renames them on the way into the
          # JAX proto, which spells them by axis.
          f'      coarsest_subgrid_shape {{ dim_0: 8 dim_1: 8 dim_2: 8 }}\n'
          f'    }}'
      )
    raise ValueError(
        f'Unrecognised solver {solver}; expected one of '
        f'{[wui_pb2.WuiCase.Solver.Name(v) for v in (0, 1, 2)]}.'
    )

  def config_text(self) -> str:
    """The solver configuration this case expands into, as a text proto.

    Returned rather than only used internally so a run can record exactly
    what it solved, and so the derivation can be checked without reading
    Python.

    Returns:
      A `SwirlLMParameters` text proto.
    """
    case = self.case
    nx = self._grid_size(self.fx, case.cores_x, 'x')
    ny = self._grid_size(self.fy, case.cores_y, 'y')
    nz = self._grid_size(self.fz, case.cores_z, 'z')

    # The lateral faces are slip walls: the normal component is Dirichlet
    # zero and the tangential ones are zero gradient. The normal component has
    # to be Dirichlet. `boundary_condition_utils.is_outflow` tests exactly one
    # thing -- all three components Neumann -- so a face left entirely zero
    # gradient is classified as an outlet, nothing constrains the transverse
    # momentum, and with `pressure_outlet` the pressure is pinned along that
    # face as well. On the cylinder that made the wake leave through the side
    # of the domain.
    #
    # The lid is the deliberate exception when `open_top` is set, which it is
    # by default. A buoyant plume rising off a fire reaches the top of the
    # domain; held there by `w = 0` it has nowhere to go but sideways, and the
    # recirculation that produces is an artefact of the box rather than
    # anything about the fire. The side faces are what the cylinder's wake
    # escaped through, and those stay walls either way.
    #
    # Closing it is worth trying when the top is far enough above the scene
    # that the plume never arrives: the power-of-two extent leaves about 148 m
    # of air above the canopy against the 100 m asked for. A lid also
    # constrains the transverse momentum an outlet leaves free, and it is the
    # better-tested of the two, every run before this one having had it.
    #
    # The floor is no-slip, but it is also buried several metres inside the
    # terrain, so nothing there is ever in the fluid.
    faces = {
        'u': [
            (0, 0, 'DIRICHLET', self.u_ref),   # inlet, overridden by bc_u_0_0
            (0, 1, 'NEUMANN', 0.0),            # outlet
            (1, 0, 'NEUMANN', 0.0),
            (1, 1, 'NEUMANN', 0.0),
            (2, 0, 'DIRICHLET', 0.0),
            (2, 1, 'NEUMANN', 0.0),
        ],
        'v': [
            (0, 0, 'DIRICHLET', 0.0),
            (0, 1, 'NEUMANN', 0.0),
            (1, 0, 'DIRICHLET', 0.0),          # slip wall: no through-flow
            (1, 1, 'DIRICHLET', 0.0),
            (2, 0, 'DIRICHLET', 0.0),
            (2, 1, 'NEUMANN', 0.0),
        ],
        'w': [
            (0, 0, 'DIRICHLET', 0.0),
            (0, 1, 'NEUMANN', 0.0),
            (1, 0, 'NEUMANN', 0.0),
            (1, 1, 'NEUMANN', 0.0),
            (2, 0, 'DIRICHLET', 0.0),
            # An outlet needs every component zero gradient; a lid needs the
            # normal one pinned. Only `w` differs between the two.
            (2, 1, 'NEUMANN' if case.open_top else 'DIRICHLET', 0.0),
        ],
    }

    def velocity_bc(name):
      body = '\n'.join(
          f'  boundary_info {{ dim: {dim} location: {location} '
          f'type: BC_TYPE_{kind} value: {value:g} }}'
          for dim, location, kind, value in faces[name]
      )
      return f'boundary_conditions {{\n  name: "{name}"\n{body}\n}}'

    pressure_faces = '\n'.join(
        f'  boundary_info {{ dim: {d} location: {l} '
        'type: BC_TYPE_NEUMANN value: 0.0 }'
        for d in range(3)
        for l in (0, 1)
    )
    pressure_bc = (
        'boundary_conditions {\n  name: "p"\n' + pressure_faces + '\n}'
    )
    variables = '\n'.join(
        f'      variables {{ name: "{name}" value: 0.0 bc: DIRICHLET }}'
        for name in VELOCITY
    )
    keys = [
        'rho_f',
        'src_u', 'src_v', 'src_w',
        'bc_u_0_0', 'bc_v_0_0', 'bc_w_0_0',
    ]
    if case.sponge_fraction > 0.0:
      # `sponge_beta` is the damping profile; the `*_init` fields are what
      # the sponge relaxes towards, which here is the undisturbed inlet.
      keys.append('sponge_beta')
      keys.extend(f'{name}_init' for name in VELOCITY)
    if case.use_sgs:
      # `velocity.prediction_step` adds the eddy viscosity to `nu` whether or
      # not anyone is tracking it, and only writes the field out when the
      # state carries this key. Declaring it is the difference between being
      # able to see what the subgrid model did and having to read the solver
      # to find out whether it ran at all.
      keys.append('nu_t')
    state_keys = '\n'.join(f'additional_state_keys: "{key}"' for key in keys)

    return f"""\
# Generated by swirl_lm.jax.examples.wui.Wui from a WuiCase.
# Scene: {case.scene_path}
# {len(self.box_centres)} building boxes on terrain, over a
# {self.lx:g} x {self.ly:g} x {self.lz:g} m domain.
solver_procedure: VARIABLE_DENSITY
convection_scheme: CONVECTION_SCHEME_QUICK
time_integration_scheme: TIME_SCHEME_CN_EXPLICIT_ITERATION
grid_params {{
  computation_shape {{
    dim_0: {case.cores_x} dim_1: {case.cores_y} dim_2: {case.cores_z}
  }}
  grid_size {{ dim_0: {nx} dim_1: {ny} dim_2: {nz} }}
  length {{ dim_0: {self.lx:g} dim_1: {self.ly:g} dim_2: {self.lz:g} }}
  halo_width: {case.halo_width}
  dt: {self.dt:.8g}
  kernel_size: {case.kernel_size}
  # Nothing is periodic. The terrain is not periodic in any direction, so a
  # periodic face would put a cliff at the seam.
  periodic {{ dim_0: false dim_1: false dim_2: false }}
}}
pressure {{
  solver {{
{self._solver_text()}
  }}
  # Pins the pressure level at every face classified as an outlet -- x-high
  # always, and the top as well when `open_top` is set. Without this each
  # non-periodic face carries a homogeneous Neumann condition, the Poisson
  # problem is singular, and the unconverged residual acts as a spurious
  # global pressure gradient.
  pressure_outlet: true
  # The density is constant, so `drho` is identically zero and filtering it
  # is pure waste -- at run time, and at compile time where XLA constant
  # folds the stencil over the whole grid. See MEMORY.md.
  num_d_rho_filter: 0
}}
thermodynamics {{ constant_density {{}} }}
density: {case.density:g}
kinematic_viscosity: {case.kinematic_viscosity:.8g}
num_sub_iterations: {case.num_sub_iterations}
use_sgs: {str(case.use_sgs).lower()}
enable_rhie_chow_correction: false
{state_keys}
boundary_models {{
  ib {{
    ghost_point {{
{variables}
{self._geometry_text()}
      num_sweeps: {case.num_sweeps}
      # Off, deliberately. The cells below the ghost layer are not part of
      # the solution and do drift -- measured, the deep interior reaches
      # 13 m/s within a few steps -- but holding them at the boundary value
      # is worse than letting them drift. The solver integrates momentum in
      # solid cells too, so a clamped cell feeds its neighbour on the next
      # step and the step in the field marches out to the fluid, where the
      # five-point diffusion stencil reads it. Measured on flat ground under
      # a 5 m/s wind: clamped, the flow was clean for 25 steps, five times
      # too fast by 100 and non-finite by 200; unclamped, it held the inflow
      # value exactly for 200. Left alone the solid stays smooth, which is
      # what the wide stencils need. See MEMORY.md.
      clamp_interior: false
      write_interior_mask: true
      write_ghost_mask: true
      write_signed_distance: false
    }}
  }}
{self._sponge_config()}
}}
{velocity_bc('u')}
{velocity_bc('v')}
{velocity_bc('w')}
{pressure_bc}
"""

  def _build_params(self) -> parameters_lib.SwirlLMParameters:
    """Expands the case into a full solver configuration."""
    config = parameters_pb2.SwirlLMParameters()
    text_format.Parse(self.config_text(), config)
    return parameters_lib.SwirlLMParameters(config)

  # -- Initialization -------------------------------------------------------

  def ghost_points(self, mesh) -> Any:
    """Builds the ghost-point geometry for `mesh`, once.

    Args:
      mesh: The device mesh.

    Returns:
      The `GhostPointGeometry` for this case.
    """
    if self._ghost_points is None:
      config = self.params.boundary_models.ib.ghost_point
      self._ghost_points = immersed_boundary_ghost_point.GhostPointGeometry(
          self.geometry_fn,
          self.gp,
          mesh,
          write_interior_mask=config.write_interior_mask,
          write_ghost_mask=config.write_ghost_mask,
          write_signed_distance=config.write_signed_distance,
      )
      self.params.additional_state_partition_specs.update(
          self._ghost_points.partition_specs()
      )
      logging.info('%s', self._ghost_points.summary())
      self._check_reach(self._ghost_points)
    return self._ghost_points

  def _check_reach(self, geometry: Any) -> None:
    """Checks the interpolation stencil against the halo on the split axes.

    An image point sits up to `reach` cells from its ghost cell, so its
    trilinear stencil reaches `ceil(reach)` cells, and across a device seam
    those cells live in the halo. A halo too narrow to hold them does not
    fail: the stencil quietly reads whatever the halo exchange last left
    there, which is the same class of error as the stale-halo bug that cost
    7% on the cylinder. On an unsplit axis there is no seam and the probe is
    capped at the array edge instead, so this only applies where the domain
    is actually divided.

    Args:
      geometry: The built `GhostPointGeometry`.

    Raises:
      ValueError: If a split axis needs a wider halo than it has.
    """
    halo = self.case.halo_width
    cores = self.gp.to_data_axis_order(
        self.case.cores_x, self.case.cores_y, self.case.cores_z
    )
    names = self.gp.data_axis_order
    for axis in range(3):
      if cores[axis] == 1:
        continue
      reach = max(g.max_reach_cells[axis] for g in geometry.per_replica)
      needed = int(math.ceil(reach))
      if needed > halo:
        raise ValueError(
            f'The image-point stencil reaches {reach:.2f} cells along '
            f'{names[axis]}, so it needs a halo of {needed}, but '
            f'`halo_width` is {halo} and that axis is split across '
            f'{cores[axis]} cores. Raise `halo_width` to {needed}, or run '
            f'{names[axis]} on one core.'
        )

  def fallback_fraction(self, geometry: Any) -> float:
    """The share of ghost cells on the nearest-fluid fallback.

    Worth looking at before a run: a fallback cell is first-order where the
    rest are second-order, and a large share means the geometry and the grid
    are not compatible -- a surface sitting on the cell centres, or terrain
    too steep for the vertical spacing.

    Args:
      geometry: The built `GhostPointGeometry`.

    Returns:
      The fraction in `[0, 1]`.
    """
    total = sum(g.num_ghost_points for g in geometry.per_replica)
    fallback = sum(g.num_fallback for g in geometry.per_replica)
    return fallback / max(total, 1)

  def _mesh_fields(
      self,
      params: parameters_lib.SwirlLMParameters,
      logical_coordinates: tuple[int, int, int],
  ) -> tuple[Any, Any, Any]:
    """This device's cell-centre coordinates, halos included.

    Taken from `grid_params.global_xyz_with_halos`, which is the same source
    the ghost-point precompute uses, so the initial condition and the
    geometry are evaluated at identical positions.

    Args:
      params: Simulation parameters.
      logical_coordinates: `(cx_i, cy_i, cz_i)` position of this device.

    Returns:
      `(xx, yy, zz)` meshgrids in data-array index order.
    """
    grid = params.grid_params
    core = grid.to_data_axis_order(
        grid.core_nx, grid.core_ny, grid.core_nz
    )
    extent = grid.to_data_axis_order(grid.nx, grid.ny, grid.nz)
    replica = grid.to_data_axis_order(*logical_coordinates)

    axes = []
    for axis in range(3):
      full = np.asarray(grid.global_xyz_with_halos[axis])
      start = replica[axis] * core[axis]
      axes.append(jnp.asarray(full[start : start + extent[axis]]))
    grids = jnp.meshgrid(*axes, indexing='ij')
    return grid.to_xyz_order(grids)

  def initialization(
      self,
      params: parameters_lib.SwirlLMParameters,
      logical_coordinates: tuple[int, int, int],
  ) -> dict[str, ScalarField]:
    """The log-law wind, the fuel field and the immersed boundary geometry.

    Args:
      params: Simulation parameters.
      logical_coordinates: `(cx_i, cy_i, cz_i)` position of this device.

    Returns:
      Dict of device-local initial states.
    """
    grid = params.grid_params
    halo = grid.halo_width
    xx, yy, zz = self._mesh_fields(params, logical_coordinates)

    terrain = self.terrain_profile()
    speed = self.inflow_profile(zz - terrain(xx, yy))
    zeros = jnp.zeros_like(speed)

    states = {
        'rho': jnp.full_like(speed, self.case.density),
        'u': speed,
        'v': zeros,
        'w': zeros,
        'p': zeros,
        'rho_f': self._fuel_field(xx, yy, zz),
    }
    for name in VELOCITY:
      states[f'src_{name}'] = jnp.zeros_like(speed)
    if self.case.use_sgs:
      states['nu_t'] = jnp.zeros_like(speed)

    if self.sponge is not None:
      # The damping profile, zero outside the layer and rising as sin^2 to
      # `a_coeff` at the face (Klemp and Lilly, 1978), and the targets the
      # sponge relaxes towards -- the undisturbed inlet, frozen at step 0.
      beta = rayleigh_damping_layer.klemp_lilly_relaxation_coeff_fn(
          list(next(iter(self.params.sponge)).orientation),
          x0=0.0, y0=0.0, z0=0.0,
      )
      states['sponge_beta'] = beta(xx, yy, zz, self.lx, self.ly, self.lz)

    geometry = self.ghost_points(_mesh_for(params))
    states.update(geometry.initial_states(logical_coordinates))

    # Start the solid at rest rather than at the inflow speed, so the first
    # step does not have to undo a wind blowing inside the ground. The ghost
    # cells are overwritten by the method immediately.
    if immersed_boundary_ghost_point.INTERIOR_MASK_KEY in states:
      fluid = states[immersed_boundary_ghost_point.INTERIOR_MASK_KEY]
      states['u'] = states['u'] * fluid

    if self.sponge is not None:
      # After the mask, not before: the sponge should pull the flow back to
      # the state the run actually started from. Captured earlier it would
      # carry a wind blowing inside the solid, which the immersed boundary
      # then spends every step undoing.
      for name in VELOCITY:
        states[f'{name}_init'] = states[name]

    # The inlet profile, as the first `halo + 1` planes of the initial field.
    # Supplying these overrides the scalar Dirichlet value in the
    # configuration, which is the only way a non-uniform inflow can be
    # expressed; `distribute_bc_state` takes them from the replicas on the
    # face and ignores the rest.
    axis = grid.get_axis_index('x')
    inlet = [slice(None)] * 3
    inlet[axis] = slice(None, halo + 1)
    for name in VELOCITY:
      states[f'bc_{name}_0_0'] = states[name][tuple(inlet)]
    return states

  def _fuel_field(self, xx: Any, yy: Any, zz: Any) -> Any:
    """Vegetation bulk density at the given positions, by nearest cell.

    Nearest neighbour rather than interpolation because the field is a
    piecewise-constant voxel density, and smoothing it would invent fuel in
    cells the source says are empty.

    Args:
      xx: Eastings in the solver frame.
      yy: Northings in the solver frame.
      zz: Heights above the domain floor.

    Returns:
      Bulk density in kg/m^3, zero outside the table.
    """
    table = jnp.asarray(self.fuel)
    index = []
    for coordinate, spacing, count in (
        (xx, self.dx, self.fx), (yy, self.dy, self.fy), (zz, self.dz, self.fz)
    ):
      raw = jnp.floor(coordinate / spacing).astype(jnp.int32)
      index.append(jnp.clip(raw, 0, count - 1))
    # Inclusive at the far edge: the last node sits exactly at `length`, and
    # a strict bound there silently left the far plane of cells with no
    # vegetation at all.
    inside = jnp.ones_like(xx, bool)
    for coordinate, length in ((xx, self.lx), (yy, self.ly), (zz, self.lz)):
      inside = inside & (coordinate >= 0.0) & (coordinate <= length)
    return jnp.where(inside, table[index[0], index[1], index[2]], 0.0)

  # -- Source terms ---------------------------------------------------------

  def source_term_update_fn(self):
    """The canopy drag, as the source-term library the driver expects.

    The form is the one the fire example uses,
    `-3/8 * c_d * rho * a_v * |u| * u`, with one difference: `a_v` is
    proportional to the local fuel density rather than uniform. A splat-free
    FastFuels canopy varies by two orders of magnitude between the tree
    crowns and the grass, and a uniform frontal area density would drag the
    sparse cells as hard as the dense ones.

    Returns:
      A dict keyed by velocity component, whose values are functions of
      `(states, additional_states)` returning that component's `src_*`.
    """
    coefficient = 3.0 * self.case.c_d * self.case.a_v_per_density / 8.0

    def drag(states, additional_states, name):
      rho_f = additional_states['rho_f']
      magnitude = jnp.sqrt(
          states['u'] ** 2 + states['v'] ** 2 + states['w'] ** 2
      )
      force = (
          -coefficient
          * states['rho']
          * rho_f
          * magnitude
          * states[name]
      )
      return jnp.where(rho_f < _EPSILON, jnp.zeros_like(force), force)

    def sponge_force(states, additional_states):
      """The Rayleigh damping, as a dict of `src_*` contributions."""
      helper = {'sponge_beta': additional_states['sponge_beta']}
      helper.update({f'src_{name}': jnp.zeros_like(states[name])
                     for name in VELOCITY})
      helper.update({f'{name}_init': additional_states[f'{name}_init']
                     for name in VELOCITY
                     if f'{name}_init' in additional_states})
      inputs = {name: states[name] for name in VELOCITY}
      inputs['rho'] = states['rho']
      return self.sponge.additional_states_update_fn(
          states=inputs, additional_states=helper, dt=self.dt
      )

    def source(name):
      def fn(states, additional_states):
        total = drag(states, additional_states, name)
        if self.sponge is not None:
          # The layer returns a primitive-variable force, so it is weighted
          # by density to match the conservative source the solver wants --
          # the same conversion the fire example makes.
          total = total + states['rho'] * sponge_force(
              states, additional_states
          )[f'src_{name}']
        return {f'src_{name}': total}

      return fn

    return {name: source(name) for name in VELOCITY}

  # -- Reporting ------------------------------------------------------------

  def summary(self) -> str:
    """A one-line description of the resolved case, for logging."""
    case = self.case
    drag = case.c_d * case.a_v_per_density * float(self.fuel.max())
    return (
        f'wui: {len(self.box_centres)} building boxes plus terrain over a '
        f'{self.lx:g} x {self.ly:g} x {self.lz:g} m domain on '
        f'{self.fx} x {self.fy} x {self.fz} points at '
        f'{self.dx:g} x {self.dy:g} x {self.dz:g} m; '
        f'u = {self.u_ref:g} m/s at {case.reference_height:g} m rising to '
        f'{self.u_max:.2f} at the lid, z0 = {case.roughness_length:g} m; '
        f'canopy c_d*a_v up to '
        f'{drag:.3g} 1/m; dt = {self.dt:.4g} s on '
        f'{case.cores_x}x{case.cores_y}x{case.cores_z} cores'
        + (
            ''
            if self.sponge is None
            else f'; sponge over the outer {case.sponge_fraction:.0%} of '
            f'the outlet'
            + (
                ''
                if case.lateral_sponge_fraction <= 0.0
                else f' and {case.lateral_sponge_fraction:.0%} of each side'
            )
            + f', relaxing in {case.sponge_coeff:g} s'
        )
    )


def _mesh_for(params: parameters_lib.SwirlLMParameters):
  """The device mesh for `params`, imported lazily to avoid a cycle."""
  from swirl_lm.jax.core import driver as driver_lib  # pylint: disable=g-import-not-at-top

  return driver_lib.create_mesh(params)
