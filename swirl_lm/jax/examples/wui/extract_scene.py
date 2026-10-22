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
"""Turns a scanned wildland-urban-interface scene into solver inputs.

Reads two datasets of the same ground and writes one compact scene file:

  * a sparse voxel classification (`.npz`) derived from Gaussian splats, which
    supplies the **buildings**;
  * a FastFuels grid (`.zarr`), which supplies the **terrain** and the
    **vegetation bulk density**.

Each source is used for what it is good at, which matters because they do not
agree with each other in the obvious way. Three properties of the data drove
the design, all measured rather than assumed:

**The two files differ by a vertical datum.** Sampling ground-classified
voxels against the FastFuels elevation gives a near-constant offset of -37.1 m
(sd 2.96 m), consistent with one being ellipsoidal height and the other
orthometric; the geoid separation in southern California is about -35 m. The
CRS labels disagree too (the grid says EPSG:26911, the voxels say 32611),
though the horizontal origins agree to 0.5 m. **Nothing here mixes absolute
elevations between the files.** Building heights come from the voxel file's
own `hag_m`, height above ground; the terrain surface comes from the grid; the
fuel is already terrain-following. Everything is height above ground, with one
vertical reference.

**The buildings are shells.** Splat-derived classes have no interior, and
roofs seen from one angle have holes, so nothing may depend on a watertight
surface. Footprints are therefore closed in **plan view**, where a hole in a
roof is invisible, and each building becomes a small union of oriented boxes
whose signed distance is exact.

**The terrain is steep.** 91 m of relief over 220 m: median slope 19 degrees,
90th percentile 35. A height-field level set reports the vertical gap rather
than the true distance, which degrades once the slope exceeds `dz/dx`. At
2 x 2 x 1 m the ghost-point precompute reports no shortened probes and no
fallback cells on this terrain, so that resolution is safe; at 2 x 2 x 2 m a
third of the ghost cells fall back. Use a vertical spacing of 1 m.

Usage:

    python -m swirl_lm.jax.examples.wui.extract_scene \
        --voxels=.../combustion_grid_0p25m.npz \
        --grid=.../mandeville_grid_data.zarr \
        --output=.../wui_scene.npz \
        --overlay=.../fit_check.png

`--centre_x/--centre_y/--half_width` crop a square sub-domain, which is worth
using while iterating even though the whole scene is only a few million cells.

The overlay is not optional in spirit: the box fitting is the step where a
quiet error costs a whole run, so look at it before running the solver.
"""

import dataclasses
import json
from typing import Any

from absl import app
from absl import flags
from absl import logging
import numpy as np
from scipy import ndimage
import zarr


_VOXELS = flags.DEFINE_string('voxels', None, 'Sparse voxel .npz.')
_GRID = flags.DEFINE_string('grid', None, 'FastFuels .zarr.')
_OUTPUT = flags.DEFINE_string('output', None, 'Scene file to write.')
_OVERLAY = flags.DEFINE_string(
    'overlay', None, 'Optional PNG comparing fitted boxes with footprints.'
)
_CENTRE_X = flags.DEFINE_float('centre_x', None, 'Crop centre easting (UTM).')
_CENTRE_Y = flags.DEFINE_float('centre_y', None, 'Crop centre northing (UTM).')
_HALF_WIDTH = flags.DEFINE_float(
    'half_width', None, 'Half width of a square crop, in metres.'
)
_DX = flags.DEFINE_float('dx', 2.0, 'Target horizontal spacing, metres.')
_DZ = flags.DEFINE_float('dz', 1.0, 'Target vertical spacing, metres.')
_AIR_ABOVE = flags.DEFINE_float(
    'air_above', 80.0,
    'Air column above the highest terrain-plus-canopy, in metres.',
)
_TARGET_SHAPE = flags.DEFINE_list(
    'target_shape', None,
    'Three integers, the exact cell counts the domain must end up with. The '
    'padding grows to reach them, so this is how a mesh is sized for a solver '
    'that constrains its extents -- multigrid wants a power of two per core. '
    'Refused if reaching it would leave less than `--margin` of clear fluid '
    'per face, or if the vertical count is under what `--air_above` needs. '
    'Left unset, the extent is whatever the crop plus the margin comes to.',
)
_ROTATE_180 = flags.DEFINE_bool(
    'rotate_180', False,
    'Rotate the scene half a turn about the vertical axis, so what was the '
    'eastern face becomes the western one. The solver always blows along +x, '
    'so this is how a wind from the east is expressed -- there is no '
    'wind-direction input anywhere downstream.',
)
_MARGIN = flags.DEFINE_float(
    'margin', 16.0,
    'Clear fluid added around the built area, in metres. A building that '
    'touches a domain face has the face boundary condition and the immersed '
    'boundary contending for the same cells, and the run diverges within ten '
    'steps -- measured. The terrain is extended outwards at its edge value '
    'and the margin carries no fuel.',
)
_MAX_BOXES = flags.DEFINE_integer(
    'max_boxes', 1,
    'Most boxes allowed per building. One, not three, because splitting a '
    'footprint measurably hurts: the parts are bounding boxes of disjoint '
    'point sets, not abutting pieces, so they leave a one or two cell sliver '
    'of fluid inside the house -- 21 of 63 boxes did on this scene. Closing '
    'those slivers is worse still, since inflating the parts to meet creates '
    'more concave creases than it removes. Measured, ghost cells with no '
    'fluid at all in their interpolation stencil: 1 041 at three boxes with '
    'one-cell parts, 885 at three boxes with the slivers closed, and 233 at '
    'one box per building. The cost is a blockier fit, median solidity 0.72 '
    'against 0.83.',
)
_MIN_AREA = flags.DEFINE_float(
    'min_footprint_area', 40.0,
    'Footprints smaller than this, in square metres, are discarded as debris.',
)
_MIN_BOX_CELLS = flags.DEFINE_integer(
    'min_box_cells', 3,
    'Narrowest box the fit may produce, in cells of the solver grid. A body '
    'under three cells wide has no cell that is not on its own surface, so '
    'the ghost-point interpolation has nothing interior to stand on. '
    'Measured on this scene at one cell: 29 of 70 boxes had no interior and '
    '1 041 ghost cells ended up with no fluid at all in their stencil.',
)
_CLOSE_GAPS_CELLS = flags.DEFINE_integer(
    'close_gaps_under_cells', 3,
    'Alleys narrower than this, in cells, are closed by growing the two '
    'boxes until they meet. A gap of one or two cells is not flow the grid '
    'can carry, but it is enough to put ghost cells in it whose stencils '
    'reach into both buildings. Closing it makes the pair one solid, which '
    'the grid can represent; the error is at most this many cells. Boxes '
    'that already touch or overlap are left alone -- their union is one '
    'body and causes no trouble. Set to 0 to leave every alley as fitted.',
)
_BUILDINGS = flags.DEFINE_enum(
    'buildings', 'fitted', ['fitted', 'single', 'none'],
    'Which buildings to put in the scene. `fitted` boxes the real roofs. '
    '`single` discards them and places one well-resolved block on the '
    'flattest ground near the centre, which isolates the building from the '
    'two things the fitted geometry confounds it with -- boxes too thin for '
    'the grid, and gaps between neighbours that are equally unresolved. '
    '`none` is the terrain-only control.',
)
_SINGLE_CELLS = flags.DEFINE_integer(
    'single_building_cells', 8,
    'Size of the `single` building, in cells on every axis: 16 x 16 m by 8 m '
    'at the default spacing. Below 3 a box has no interior at all. Six is '
    'viable but, rotated to the wind, comes out 81% surface; eight leaves a '
    'two-to-one ghost-to-interior ratio.',
)
_SINGLE_YAW = flags.DEFINE_float(
    'single_building_yaw', 45.0,
    'Rotation of the `single` building about z, in degrees from the wind, '
    'which blows along +x. 45 is the worst case for a Cartesian grid and so '
    'the honest one to test.',
)
_SINGLE_X = flags.DEFINE_float(
    'single_building_x', None,
    'Where to put the `single` building, in metres from the domain corner. '
    'Defaults to the flattest window near the centre.',
)
_SINGLE_Y = flags.DEFINE_float('single_building_y', None, 'See '
                               '--single_building_x.')

# Footprints come from **roofs alone**. Measured on the Mandeville scene,
# including `wall` and `other_builtup` is actively wrong: driveways, patios
# and retaining walls bridge neighbouring properties, and a single closing
# pass then merges the lot into one component covering 28% of the domain.
# Roofs alone, closed at 0.5 m, give 24 components over 40 m^2 -- exactly the
# 24 houses in the scene. Walls hang below the roof edge and are inside the
# box anyway; poles and vehicles are excluded as small and numerous.
BUILDING_CLASSES = (5,)             # roof
# Used only for the datum check.
GROUND_CLASSES = (2, 7, 8, 10)      # grass, asphalt, dirt, water

# A roof's height is taken at this quantile of its voxels rather than at the
# maximum, which on splat data is an antenna or a reconstruction spike.
_ROOF_QUANTILE = 95.0
# Plan-view closing radius. 0.5 m bridges the gaps between roof voxels without
# bridging the gaps between houses; at 1.0 m neighbouring roofs start to merge.
_CLOSING_RADIUS_M = 0.5
# How far to sink each box below the lowest ground under its footprint. The
# geometry is a union, so burying is free, and it removes any seam where a
# building meets sloping ground.
_BURY_DEPTH_M = 3.0
# A fitted box is accepted when this fraction of its area is really building.
_FILL_TARGET = 0.90
# Cell count each axis is rounded to. Every per-core extent, halos included,
# must divide by the kernel widths, which makes a multiple of four the real
# requirement (MEMORY.md); rounding to eight instead costs at most 16 m of
# domain and keeps the useful decompositions legal up to eight cores on an
# axis. `Wui` re-checks this against the decomposition actually requested.
_CELL_MULTIPLE = 8


@dataclasses.dataclass
class Voxels:
  """The sparse voxel classification."""

  ijk: np.ndarray
  class_id: np.ndarray
  hag_m: np.ndarray
  origin: np.ndarray
  voxel_size: float
  shape: np.ndarray
  class_names: list[str]

  def xy(self) -> tuple[np.ndarray, np.ndarray]:
    """Voxel centre eastings and northings."""
    return (
        self.origin[0] + (self.ijk[:, 0] + 0.5) * self.voxel_size,
        self.origin[1] + (self.ijk[:, 1] + 0.5) * self.voxel_size,
    )


@dataclasses.dataclass
class Grid:
  """The FastFuels grid: terrain and fuel on a regular horizontal mesh."""

  elevation: np.ndarray          # (ny, nx), row 0 is north
  bulk_density: np.ndarray       # (nz, ny, nx), terrain-following, 1 m levels
  surface_load: np.ndarray       # (ny, nx), kg/m^2, zero where absent
  surface_depth: np.ndarray      # (ny, nx), m, the layer the load occupies
  west: float
  south: float
  north: float
  dx: float
  dz: float

  def elevation_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nearest-cell elevation at eastings `x` and northings `y`.

    Row 0 of the array is the **north** edge -- the affine has a negative dy.
    That was confirmed against the voxel file: sampling this way leaves a
    2.96 m scatter against the ground voxels, and sampling the other way
    leaves 13.95 m.

    Args:
      x: Eastings.
      y: Northings.

    Returns:
      Elevation at each point.
    """
    col = np.clip(
        ((x - self.west) / self.dx).astype(int), 0, self.elevation.shape[1] - 1
    )
    row = np.clip(
        ((self.north - y) / self.dx).astype(int), 0, self.elevation.shape[0] - 1
    )
    return self.elevation[row, col]


def read_voxels(path: str) -> Voxels:
  """Loads the sparse voxel classification."""
  with np.load(path, allow_pickle=False) as archive:
    needed = ('ijk', 'class_id', 'hag_m', 'grid_origin_utm', 'voxel_size_m',
              'grid_shape', 'class_names')
    missing = [k for k in needed if k not in archive.files]
    if missing:
      raise ValueError(f'{path} is missing {missing}.')
    voxels = Voxels(
        ijk=np.asarray(archive['ijk']),
        class_id=np.asarray(archive['class_id']),
        hag_m=np.asarray(archive['hag_m'], np.float64),
        origin=np.asarray(archive['grid_origin_utm'], np.float64),
        voxel_size=float(np.asarray(archive['voxel_size_m']).item()),
        shape=np.asarray(archive['grid_shape']),
        class_names=[str(s) for s in np.asarray(archive['class_names'])],
    )
  logging.info(
      'voxels: %.1f M occupied, %.2f m, extent %.0f x %.0f x %.0f m',
      len(voxels.ijk) / 1e6, voxels.voxel_size,
      *(voxels.shape * voxels.voxel_size),
  )
  return voxels


def read_grid(path: str) -> Grid:
  """Loads terrain and fuel from the FastFuels store."""
  store = zarr.open(path, mode='r')
  attrs = dict(store.attrs)
  bounds = attrs['bounds']
  # The surface layer is a later addition to the FastFuels product: the
  # Mandeville store this case was built against has `tree` and `topography`
  # only. Absent, the surface contributes nothing and the canopy alone is the
  # fuel, which is what every run before this one used.
  if 'surface' in store:
    surface_load = np.asarray(store['surface/fuelLoad']).astype(np.float64)
    surface_depth = np.asarray(store['surface/fuelDepth']).astype(np.float64)
  else:
    logging.warning(
        '%s has no `surface` group; the surface fuel load is taken as zero '
        'and only the canopy contributes', path,
    )
    shape = np.asarray(store['topography/elevation']).shape
    surface_load = np.zeros(shape, np.float64)
    surface_depth = np.zeros(shape, np.float64)
  grid = Grid(
      elevation=np.asarray(store['topography/elevation']).astype(np.float64),
      bulk_density=np.asarray(store['tree/bulkDensity']).astype(np.float32),
      surface_load=surface_load,
      surface_depth=surface_depth,
      west=float(bounds['west']),
      south=float(bounds['south']),
      north=float(bounds['north']),
      dx=float(attrs['dx']),
      dz=float(attrs['dz']),
  )
  logging.info(
      'grid: elevation %.0f..%.0f m (relief %.0f), canopy %s, %.1f%% non-zero',
      grid.elevation.min(), grid.elevation.max(),
      grid.elevation.max() - grid.elevation.min(),
      grid.bulk_density.shape, 100.0 * (grid.bulk_density > 0).mean(),
  )
  present = grid.surface_depth > 0.0
  if present.any():
    density = grid.surface_load[present] / grid.surface_depth[present]
    logging.info(
        'surface fuel: %.1f%% of cells carry a load, depth %.2f..%.2f m, '
        'bulk density %.3f..%.3f kg/m^3 (median %.3f)',
        100.0 * present.mean(),
        grid.surface_depth[present].min(), grid.surface_depth[present].max(),
        density.min(), density.max(), float(np.median(density)),
    )
  return grid


def report_datum_offset(voxels: Voxels, grid: Grid) -> float:
  """Measures the vertical disagreement between the two files.

  Reported rather than corrected. Nothing downstream mixes absolute
  elevations, so the offset does not need removing -- but if it ever drifts
  away from a constant, the two files have stopped describing the same ground
  and that must not pass silently.

  Args:
    voxels: The voxel classification.
    grid: The FastFuels grid.

  Returns:
    The median offset in metres.
  """
  ground = (voxels.hag_m < 0.3) & np.isin(voxels.class_id, GROUND_CLASSES)
  if not ground.any():
    logging.warning('no ground-classified voxels; skipping the datum check')
    return 0.0
  x, y = voxels.xy()
  z = voxels.origin[2] + (voxels.ijk[:, 2] + 0.5) * voxels.voxel_size
  delta = z[ground] - grid.elevation_at(x[ground], y[ground])
  median, spread = float(np.median(delta)), float(delta.std())
  logging.info(
      'vertical datum offset (voxels minus grid): %+.2f m, sd %.2f m '
      '-- consistent with ellipsoidal versus orthometric height',
      median, spread,
  )
  if spread > 8.0:
    logging.error(
        'the offset is not constant (sd %.1f m): the two files may not '
        'describe the same ground, or the row order is wrong', spread,
    )
  return median


def _wall_aligned_yaw(points: np.ndarray) -> float:
  """The rotation that minimises the axis-aligned bounding area.

  Buildings have orthogonal walls, so the frame that boxes the footprint most
  tightly is the frame aligned with the walls. The obvious alternative,
  principal component analysis, is worse: on an L-shaped footprint it returns
  the diagonal, where no axis-aligned cut lines up with anything. Measured on
  a synthetic L, PCA gives 28.5 degrees and this gives 2.5, and the fitted
  boxes go from 68% to 90% real building.

  Args:
    points: `(n, 2)` footprint coordinates.

  Returns:
    The yaw in degrees, in [0, 90).
  """
  hull = points
  if len(points) > 4000:                 # the extremes decide the answer
    hull = points[np.random.default_rng(0).choice(len(points), 4000, False)]
  best_area, best_yaw = np.inf, 0.0
  for yaw in np.arange(0.0, 90.0, 0.5):
    local = _to_frame(hull, yaw)
    area = float(np.prod(local.max(0) - local.min(0)))
    if area < best_area:
      best_area, best_yaw = area, float(yaw)
  return best_yaw


def _to_frame(points: np.ndarray, yaw: float) -> np.ndarray:
  """Rotates points into the frame at `yaw` degrees."""
  r = np.radians(yaw)
  rotation = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
  return points @ rotation.T


def _from_frame(points: np.ndarray, yaw: float) -> np.ndarray:
  """The inverse of `_to_frame`."""
  r = np.radians(yaw)
  rotation = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
  return points @ rotation


def _split(
    local: np.ndarray, max_boxes: int, cell_area: float, min_width: float
) -> list[tuple]:
  """Bisects the footprint until each part is mostly solid.

  Repeatedly takes whichever part wastes the most area and cuts it across its
  longer side, at the position that leaves the two halves fullest. One box
  suits a rectangle, two an L; a U needs more than three and will come out
  loose, which the fit report makes visible.

  Args:
    local: `(n, 2)` footprint coordinates in the wall-aligned frame, metres.
    max_boxes: Most parts to produce.
    cell_area: Ground area of one footprint pixel, in square metres. The fill
      test compares a count of pixels against an area, so without this it is
      wrong by `1/cell_area` -- 16x at 0.25 m, which passes every time and
      silently leaves every building as a single box.
    min_width: Narrowest part to produce, in metres. A cut that would leave
      a sliver is skipped instead. Without this the splitter happily shaves
      a 0.5 m strip off a house, which the solver grid cannot represent at
      all: on the Mandeville scene 7 of 70 boxes came out narrower than one
      cell.

  Returns:
    A list of `(low, high)` corner pairs.
  """
  parts = [local]
  while len(parts) < max_boxes:
    waste = [
        float(np.prod(p.max(0) - p.min(0))) - len(p) * cell_area
        for p in parts
    ]
    index = int(np.argmax(waste))
    points = parts[index]
    low, high = points.min(0), points.max(0)
    fill = len(points) * cell_area / max(float(np.prod(high - low)), 1e-9)
    if fill > _FILL_TARGET:
      break
    axis = int(np.argmax(high - low))
    best_score, best_cut = -np.inf, None
    for fraction in np.linspace(0.15, 0.85, 40):
      cut = low[axis] + fraction * (high[axis] - low[axis])
      if cut - low[axis] < min_width or high[axis] - cut < min_width:
        continue
      left, right = points[points[:, axis] <= cut], points[points[:, axis] > cut]
      if len(left) < 20 or len(right) < 20:
        continue
      score = sum(
          len(p) ** 2 * cell_area / max(float(np.prod(p.max(0) - p.min(0))), 1e-9)
          for p in (left, right)
      ) / len(points)
      if score > best_score:
        best_score, best_cut = score, cut
    if best_cut is None:
      break
    left = points[points[:, axis] <= best_cut]
    right = points[points[:, axis] > best_cut]
    parts = parts[:index] + [left, right] + parts[index + 1:]
  return [(p.min(0), p.max(0)) for p in parts]


def _solidity(
    local: np.ndarray, boxes: list[tuple], cell_area: float
) -> float:
  """The fraction of the boxed area that is really building.

  Only this is reported. The obvious companion metric, the fraction of the
  footprint covered by the boxes, is structurally 1 here -- every box is the
  bounding box of the points assigned to it -- so quoting it would imply
  information it does not carry.

  Args:
    local: `(n, 2)` footprint coordinates in the box frame, in metres.
    boxes: `(low, high)` corner pairs, in metres.
    cell_area: Ground area of one footprint pixel, in square metres. Without
      this the count of pixels is compared against an area in square metres
      and the answer comes out scaled by `1/cell_area` -- 16x at 0.25 m.

  Returns:
    Solidity in [0, 1].
  """
  area = sum(float(np.prod(high - low)) for low, high in boxes)
  return float(len(local) * cell_area / max(area, 1e-9))


def extract_buildings(
    voxels: Voxels,
    grid: Grid,
    crop: tuple[float, float, float, float],
    max_boxes: int,
    min_area_m2: float,
    min_width: float,
    min_gap: float,
    z_floor: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
  """Fits oriented boxes to every building footprint.

  The footprint is closed and filled **in plan view**, which is what makes
  the shell-only input usable: a hole in a roof does not show up in the
  projection, so no three-dimensional repair is needed.

  Args:
    voxels: The voxel classification.
    grid: The FastFuels grid, for the ground under each footprint.
    crop: `(west, south, east, north)` of the domain, in UTM.
    max_boxes: Most boxes per building.
    min_area_m2: Footprints below this are discarded as debris.
    min_width: Narrowest box to produce, in metres.
    min_gap: Alleys narrower than this, in metres, are closed by growing
      the boxes on either side until they meet.
    z_floor: Elevation of the bottom of the solver domain, which the box
      centres are measured from.

  Returns:
    Arrays ready for `signed_distance.batched_boxes`, in the solver frame --
    metres from the south-west corner of the domain floor -- and a
    per-building report.
  """
  west, south, east, north = crop
  building = np.isin(voxels.class_id, BUILDING_CLASSES)
  x, y = voxels.xy()
  inside = (x >= west) & (x < east) & (y >= south) & (y < north)
  keep = building & inside
  if not keep.any():
    raise ValueError('no building voxels inside the requested crop')

  size = voxels.voxel_size
  nx = int(np.ceil((east - west) / size))
  ny = int(np.ceil((north - south) / size))
  col = np.clip(((x[keep] - west) / size).astype(int), 0, nx - 1)
  row = np.clip(((y[keep] - south) / size).astype(int), 0, ny - 1)

  footprint = np.zeros((ny, nx), bool)
  footprint[row, col] = True
  # Close gaps between wall voxels, then fill. Both in plan view.
  radius = max(1, int(round(_CLOSING_RADIUS_M / size)))
  footprint = ndimage.binary_closing(
      footprint, np.ones((2 * radius + 1, 2 * radius + 1))
  )
  footprint = ndimage.binary_fill_holes(footprint)

  labels, count = ndimage.label(footprint)
  min_pixels = int(min_area_m2 / size**2)
  logging.info(
      '%d footprint components before pruning below %.0f m^2',
      count, min_area_m2,
  )

  hag = voxels.hag_m[keep]
  centres, half_extents, rotations, owners = [], [], [], []
  report: list[dict[str, Any]] = []

  for label in range(1, count + 1):
    pixels = np.argwhere(labels == label)        # (row, col)
    if len(pixels) < min_pixels:
      continue
    points = np.stack(
        [west + (pixels[:, 1] + 0.5) * size, south + (pixels[:, 0] + 0.5) * size],
        axis=1,
    )
    yaw = _wall_aligned_yaw(points)
    local = _to_frame(points, yaw)
    boxes = _split(local, max_boxes, size**2, min_width)
    solidity = _solidity(local, boxes, size**2)

    # Roof height from this component's own voxels, at a quantile rather than
    # the maximum so a single splat spike cannot stretch the building.
    member = (labels[row, col] == label)
    heights = hag[member]
    if heights.size == 0:
      continue
    roof = float(np.percentile(heights, _ROOF_QUANTILE))

    ground = grid.elevation_at(points[:, 0], points[:, 1])
    base = float(ground.min()) - _BURY_DEPTH_M
    top = float(np.median(ground)) + roof

    for low, high in boxes:
      centre_local = 0.5 * (low + high)
      centre_xy = _from_frame(centre_local[None, :], yaw)[0]
      centres.append([centre_xy[0] - west, centre_xy[1] - south,
                      0.5 * (base + top) - z_floor])
      # Never narrower than one solver cell. A body the grid cannot resolve
      # is worse than one that is slightly too fat: it produces ghost cells
      # whose interpolation stencils reach straight through the body.
      half = 0.5 * (high - low)
      half_extents.append([max(half[0], 0.5 * min_width),
                           max(half[1], 0.5 * min_width),
                           0.5 * (top - base)])
      rotations.append([0.0, 0.0, yaw])
      owners.append(int(label))

    report.append({
        'label': int(label),
        'boxes': len(boxes),
        'yaw_deg': yaw,
        'roof_height_m': roof,
        'footprint_m2': float(len(pixels) * size**2),
        'solidity': solidity,
    })

  if not centres:
    raise ValueError('every footprint was pruned; check --min_footprint_area')

  half_extents = _close_narrow_gaps(
      np.asarray(centres, np.float64), np.asarray(half_extents, np.float64),
      np.asarray(rotations, np.float64), np.asarray(owners, np.int32),
      min_gap,
  )
  boxes_out = {
      'box_centres': np.asarray(centres, np.float32),
      'box_half_extents': np.asarray(half_extents, np.float32),
      'box_rotations_deg': np.asarray(rotations, np.float32),
      # Which building each box came from. Parts of one building abut by
      # construction, so a gap report that did not know this would measure
      # the splitter rather than the spacing of the houses.
      'box_building_id': np.asarray(owners, np.int32),
  }
  logging.info(
      '%d buildings -> %d boxes', len(report), len(centres)
  )
  return boxes_out, report


def _target_padding(
    have: int, want: int, minimum_cells: int, axis: str
) -> tuple[int, int]:
  """Cells to add at the low and high ends of one axis to reach `want`.

  Split as evenly as the difference allows, the odd cell going to the high
  end. An odd split is not a problem in itself -- the margin is clear fluid
  either way -- but it does move the domain centre half a cell off the crop
  centre, so it is worth being deliberate about which side gets it.

  Args:
    have: The cell count after cropping.
    want: The cell count required.
    minimum_cells: The least clear fluid each face must carry, in cells.
    axis: The axis name, for the message.

  Returns:
    `(low, high)` cell counts to pad.

  Raises:
    ValueError: If `want` is below `have`, or if reaching it would leave less
      than `minimum_cells` on a face.
  """
  if want < have:
    raise ValueError(
        f'--target_shape asks for {want} cells along {axis} but the crop is '
        f'already {have}. The crop can only be trimmed by shrinking '
        f'--half_width; the padding cannot remove cells.'
    )
  extra = want - have
  low, high = extra // 2, extra - extra // 2
  if min(low, high) < minimum_cells:
    shortfall = minimum_cells - min(low, high)
    raise ValueError(
        f'--target_shape asks for {want} cells along {axis}, which leaves '
        f'{min(low, high)} cells of clear fluid on a face where --margin '
        f'needs {minimum_cells}. Raise the target by at least '
        f'{2 * shortfall} cells, or lower --margin.'
    )
  return low, high


def _snap_crop(
    crop: tuple[float, float, float, float], dx: float
) -> tuple[float, float, float, float]:
  """Trims the crop so the horizontal cell counts are decomposable.

  Trims rather than extends, because extending would walk off the edge of the
  elevation raster and the terrain there would be an extrapolation.

  Args:
    crop: `(west, south, east, north)` in UTM.
    dx: Target horizontal spacing.

  Returns:
    The trimmed crop, with `(east - west) / dx` and `(north - south) / dx`
    both multiples of `_CELL_MULTIPLE`.
  """
  west, south, east, north = crop
  nx = int((east - west) / dx) // _CELL_MULTIPLE * _CELL_MULTIPLE
  ny = int((north - south) / dx) // _CELL_MULTIPLE * _CELL_MULTIPLE
  if nx < _CELL_MULTIPLE or ny < _CELL_MULTIPLE:
    raise ValueError(
        f'The crop is {east - west:.0f} x {north - south:.0f} m, which is '
        f'under {_CELL_MULTIPLE} cells across at dx = {dx} m.'
    )
  trimmed = (west, south, west + nx * dx, south + ny * dx)
  if east - trimmed[2] > 0.5 * dx or north - trimmed[3] > 0.5 * dx:
    logging.info(
        'trimmed %.0f m off the east edge and %.0f m off the north edge so '
        'the mesh is %d x %d cells', east - trimmed[2], north - trimmed[3],
        nx, ny,
    )
  return trimmed


def resample_terrain(
    grid: Grid, crop: tuple[float, float, float, float], dx: float
) -> np.ndarray:
  """Terrain elevation on the target horizontal mesh, above sea level.

  Interpolated bilinearly rather than sampled at the nearest cell, which
  matters only when the target is finer than the source and matters a lot
  then. The FastFuels elevation is on a 2 m mesh derived from a 10 m DEM, so
  asking for 1 m adds no information -- a mid row gains 4 distinct elevations
  out of 57 while the cell count quadruples -- but nearest-cell sampling
  turns each source cell into a block, and the result is a staircase.
  Measured on this terrain at 1 m: nearest gives a median slope of 26.6
  degrees with 35.7% of cells flat, against 19.5 and 19.7% at 2 m, because
  the plateaus widen and the step edges steepen. Bilinear gives 20.6 degrees
  and 13.2%. Steep steps are precisely what a height-field ghost boundary
  handles worst.

  At the source spacing the two agree exactly, so this changes nothing for a
  2 m scene.

  Args:
    grid: The FastFuels grid.
    crop: `(west, south, east, north)` in UTM.
    dx: Target horizontal spacing.

  Returns:
    A `(nx, ny)` array of elevations, in metres above sea level, indexed with
    x first and y increasing northward -- the solver's convention, not the
    raster's.
  """
  west, south, east, north = crop
  nx = int(round((east - west) / dx))
  ny = int(round((north - south) / dx))
  xs = west + (np.arange(nx) + 0.5) * dx
  ys = south + (np.arange(ny) + 0.5) * dx

  rows, cols = grid.elevation.shape
  # Row 0 of the raster is the north edge, hence `north - y`.
  fx = np.clip((xs - grid.west) / grid.dx - 0.5, 0.0, cols - 1.0)
  fy = np.clip((grid.north - ys) / grid.dx - 0.5, 0.0, rows - 1.0)
  i0 = np.floor(fx).astype(int)
  j0 = np.floor(fy).astype(int)
  i1 = np.minimum(i0 + 1, cols - 1)
  j1 = np.minimum(j0 + 1, rows - 1)
  tx = (fx - i0)[:, None]
  ty = (fy - j0)[None, :]
  corner = lambda j, i: grid.elevation[np.ix_(j, i)].T
  return (
      (1.0 - tx) * (1.0 - ty) * corner(j0, i0)
      + tx * (1.0 - ty) * corner(j0, i1)
      + (1.0 - tx) * ty * corner(j1, i0)
      + tx * ty * corner(j1, i1)
  )


def resample_fuel(
    grid: Grid,
    terrain: np.ndarray,
    crop: tuple[float, float, float, float],
    dx: float,
    dz: float,
    z_floor: float,
    nz: int,
) -> np.ndarray:
  """Vegetation bulk density on the target Cartesian mesh.

  The FastFuels fuel grid is **terrain-following**: level `k` sits `k` metres
  above the ground, not at a fixed elevation. Mapping it onto a Cartesian mesh
  therefore means looking up each target cell by its height above the local
  terrain, which is also why this is the one place the terrain has to be
  consistent between the two.

  Two contributions, summed. The canopy arrives already as a bulk density, one
  value per metre of height. The surface layer arrives as a **load** in kg/m^2
  over a separate depth in metres, which is what `surface/fuelDepth` is in the
  store for. Measured on the Sleepy Hollow data the layer is 0.09 to 0.46 m
  deep carrying up to 1.39 kg/m^3 of density, against a canopy reaching 3.42;
  the surface covers 91% of cells and the canopy 2%, so neither dominates
  everywhere.

  The surface is distributed by **overlap**, so that each cell holds the mean
  density of the fuel actually inside it: `load * overlap / (depth * dz)`. The
  invariant that fixes this is the column integral, because the drag on a
  column is proportional to `sum(rho_f) * dz` -- so the load, not the in-layer
  density, is the thing that must survive a change of mesh.

  Both limits come out right. A layer thinner than a cell, which at 1 m is
  every cell on this data, puts `load / dz` into the first cell above the
  ground. A cell entirely inside a thick layer gets `load / depth`, the
  density within it.

  Testing the cell *centre* against the layer instead drops the fuel
  altogether here: the first centre above ground sits half a cell up, 0.5 m at
  this spacing, and the layer is at most 0.46 m deep, so no centre is ever
  inside it.

  Args:
    grid: The FastFuels grid.
    terrain: `(nx, ny)` elevations from `resample_terrain`.
    crop: `(west, south, east, north)` in UTM.
    dx: Target horizontal spacing.
    dz: Target vertical spacing.
    z_floor: Elevation of the bottom of the solver domain.
    nz: Vertical cells.

  Returns:
    An `(nx, ny, nz)` array of bulk density in kg/m^3.
  """
  west, south, east, north = crop
  nx, ny = terrain.shape
  xs = west + (np.arange(nx) + 0.5) * dx
  ys = south + (np.arange(ny) + 0.5) * dx
  col = np.clip(((xs - grid.west) / grid.dx).astype(int), 0,
                grid.bulk_density.shape[2] - 1)
  row = np.clip(((grid.north - ys) / grid.dx).astype(int), 0,
                grid.bulk_density.shape[1] - 1)
  # (nx, ny, n_levels) column of fuel above each target column.
  column = grid.bulk_density[:, row[None, :], col[:, None]].transpose(1, 2, 0)

  z_centres = z_floor + (np.arange(nz) + 0.5) * dz
  hag = z_centres[None, None, :] - terrain[:, :, None]
  level = np.floor(hag / grid.dz).astype(int)
  valid = (level >= 0) & (level < column.shape[2])
  safe = np.clip(level, 0, column.shape[2] - 1)
  fuel = np.take_along_axis(column, safe, axis=2)
  fuel = np.where(valid, fuel, 0.0)

  # The surface layer, spread over whatever part of each cell it occupies.
  # Guarded on the depth rather than the load: a zero depth is how the store
  # says there is no surface fuel here, and dividing by it would put an
  # infinity into a field the drag reads directly.
  depth = grid.surface_depth[row[None, :], col[:, None]][:, :, None]
  load = grid.surface_load[row[None, :], col[:, None]][:, :, None]
  overlap = np.clip(
      np.minimum(hag + 0.5 * dz, depth) - np.maximum(hag - 0.5 * dz, 0.0),
      0.0,
      None,
  )
  fuel = fuel + np.where(
      depth > 0.0,
      load * overlap / np.where(depth > 0.0, depth * dz, 1.0),
      0.0,
  )

  return fuel.astype(np.float32)


def flattest_patch(
    terrain: np.ndarray, dx: float, window_m: float, search_m: float = 50.0
) -> tuple[float, float]:
  """Where to stand a test building so the terrain is not a confounder.

  Searches for the window of the given size, within `search_m` of the domain
  centre, whose mean slope and local relief are smallest. On the Mandeville
  terrain this finds a shelf at (186, 140) with 0.4 degrees of mean slope and
  no relief at all, against 3.9 degrees at the exact centre.

  Args:
    terrain: `(nx, ny)` terrain height above the domain floor.
    dx: Horizontal spacing.
    window_m: Side of the window to flatten over -- use the building size.
    search_m: How far from the centre to look.

  Returns:
    `(x, y)` in metres from the domain corner.
  """
  gx, gy = np.gradient(terrain, dx, dx)
  slope = np.degrees(np.arctan(np.hypot(gx, gy)))
  window = max(int(round(window_m / dx)), 1)
  mean_slope = ndimage.uniform_filter(slope, window)
  relief = (ndimage.maximum_filter(terrain, window)
            - ndimage.minimum_filter(terrain, window))

  nx, ny = terrain.shape
  cx, cy = nx // 2, ny // 2
  reach = max(int(round(search_m / dx)), 1)
  lo_x, hi_x = max(cx - reach, 0), min(cx + reach, nx)
  lo_y, hi_y = max(cy - reach, 0), min(cy + reach, ny)
  patch = (slice(lo_x, hi_x), slice(lo_y, hi_y))
  # Both terms are in the units they are: degrees and metres. They are not
  # commensurate, but both are near zero on good ground and the minimum is
  # insensitive to the weighting.
  i, j = np.unravel_index(
      np.argmin(mean_slope[patch] + relief[patch]),
      (hi_x - lo_x, hi_y - lo_y),
  )
  x, y = (lo_x + i) * dx, (lo_y + j) * dx
  logging.info(
      'single building at (%.0f, %.0f) m: mean slope %.1f deg, relief %.1f m '
      'over a %.0f m window (the domain centre has %.1f deg)',
      x, y, mean_slope[lo_x + i, lo_y + j], relief[lo_x + i, lo_y + j],
      window_m, mean_slope[cx, cy],
  )
  return float(x), float(y)


def single_building(
    terrain: np.ndarray,
    dx: float,
    dz: float,
    cells: int,
    yaw_deg: float,
    position: tuple[float, float] | None,
) -> dict[str, np.ndarray]:
  """One rectangular block, sized in cells rather than in metres.

  Sizing in cells is the point: the fitted geometry fails because 29 of its
  70 boxes are under three cells wide and so have no interior, and a body
  the grid cannot resolve cannot be given a sensible boundary condition.

  Buried and topped exactly as a fitted box is, so it merges with the ground
  rather than floating above it.

  Args:
    terrain: `(nx, ny)` terrain height above the domain floor.
    dx: Horizontal spacing.
    dz: Vertical spacing.
    cells: Size on every axis, in cells.
    yaw_deg: Rotation about z, in degrees.
    position: `(x, y)` in metres from the domain corner, or None to use the
      flattest window near the centre.

  Returns:
    Arrays ready for `signed_distance.batched_boxes`.

  Raises:
    ValueError: If `cells` is under 3, where the box has no interior.
  """
  if cells < 3:
    raise ValueError(
        f'`single_building_cells` is {cells}. A box under 3 cells wide has '
        'no cell that is not on its own surface, so every solid cell is a '
        'ghost cell and the interpolation has nothing interior to stand on. '
        'This is exactly the defect the fitted geometry has.'
    )
  width, height = cells * dx, cells * dz
  x, y = position if position is not None else flattest_patch(
      terrain, dx, window_m=1.5 * width
  )

  # The ground under the footprint, using the circumscribed radius so a
  # rotated box is covered whatever its yaw.
  nx, ny = terrain.shape
  xs = (np.arange(nx) + 0.0) * dx
  ys = (np.arange(ny) + 0.0) * dx
  xx, yy = np.meshgrid(xs, ys, indexing='ij')
  under = np.hypot(xx - x, yy - y) <= 0.5 * width * np.sqrt(2.0) + dx
  ground = terrain[under] if under.any() else terrain.ravel()

  base = float(ground.min()) - _BURY_DEPTH_M
  top = float(np.median(ground)) + height
  logging.info(
      'single building: %.0f x %.0f m footprint, %.0f m tall, yaw %.0f deg '
      '(%d x %d x %d cells)', width, width, height, yaw_deg, cells, cells,
      cells,
  )
  return {
      'box_centres': np.asarray(
          [[x, y, 0.5 * (base + top)]], np.float32
      ),
      'box_half_extents': np.asarray(
          [[0.5 * width, 0.5 * width, 0.5 * (top - base)]], np.float32
      ),
      'box_rotations_deg': np.asarray([[0.0, 0.0, yaw_deg]], np.float32),
      'box_building_id': np.asarray([0], np.int32),
  }


def _no_buildings() -> dict[str, np.ndarray]:
  """An empty box set, for the terrain-only control."""
  return {
      'box_centres': np.zeros((0, 3), np.float32),
      'box_half_extents': np.zeros((0, 3), np.float32),
      'box_rotations_deg': np.zeros((0, 3), np.float32),
      'box_building_id': np.zeros((0,), np.int32),
  }


def _footprint_corners(
    centre: np.ndarray, half: np.ndarray, yaw_deg: float
) -> np.ndarray:
  """The four plan-view corners of one box, in world coordinates."""
  local = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
  return _from_frame(local * half[:2], yaw_deg) + centre[:2]


def _polygon_gap(a: np.ndarray, b: np.ndarray) -> float:
  """Least distance between two convex plan-view polygons, 0 if they touch.

  Circumscribed radii will not do here. A box rotated 45 degrees has a
  circumradius 1.41 times its half width, so two houses a comfortable four
  cells apart can appear to overlap. This is the real separation.

  Args:
    a: `(n, 2)` corners of the first polygon, in order.
    b: `(m, 2)` corners of the second.

  Returns:
    The gap in metres, 0 when they intersect.
  """

  def inside(point, poly):
    # Convex, corners ordered, so the point is inside when it is on the same
    # side of every edge.
    edges = np.roll(poly, -1, axis=0) - poly
    cross = np.cross(edges, point - poly)
    return bool(np.all(cross >= -1e-9) or np.all(cross <= 1e-9))

  if any(inside(p, b) for p in a) or any(inside(p, a) for p in b):
    return 0.0

  def point_to_edges(point, poly):
    starts = poly
    ends = np.roll(poly, -1, axis=0)
    seg = ends - starts
    length2 = np.einsum('ij,ij->i', seg, seg)
    t = np.clip(
        np.einsum('ij,ij->i', point - starts, seg) / np.maximum(length2, 1e-12),
        0.0, 1.0,
    )
    nearest = starts + t[:, None] * seg
    return float(np.linalg.norm(nearest - point, axis=1).min())

  return min(
      min(point_to_edges(p, b) for p in a),
      min(point_to_edges(p, a) for p in b),
  )


def _close_narrow_gaps(
    centres: np.ndarray,
    half: np.ndarray,
    rotations: np.ndarray,
    owner: np.ndarray,
    min_gap: float,
    passes: int = 50,
) -> np.ndarray:
  """Grows boxes until any alley too narrow for the grid is closed.

  A one- or two-cell gap between two buildings is the worst case for a
  ghost-point method: too narrow to carry flow, but wide enough to hold
  ghost cells whose interpolation stencils reach into both bodies. Measured
  on the fitted Mandeville scene, 22 of 63 boxes sit in one.

  Closing the alley turns the pair into a single solid, which the grid can
  represent, at a cost of at most `min_gap` in position. Boxes that already
  touch or overlap are left alone: their union is one body and behaves.

  Applies between parts of one building as well as between buildings: a
  split produces bounding boxes of disjoint point sets, which can leave a
  sliver inside a single house.

  Only horizontal extents are grown. The vertical one is set by the roof and
  the burial depth, and two buildings do not form an alley in z.

  Args:
    centres: `(n, 3)` box centres.
    half: `(n, 3)` half extents.
    rotations: `(n, 3)` rotations in degrees.
    owner: `(n,)` which building each box belongs to.
    min_gap: Alleys narrower than this, in metres, are closed.
    passes: Most sweeps to take. Growing one box can narrow another gap, so
      this iterates to a fixed point and stops early once nothing grows.
      Measured on this scene it converges in about 25; three was not
      enough and left 11 alleys open.

  Returns:
    The half extents, with horizontal ones grown where needed.
  """
  if min_gap <= 0.0 or len(centres) < 2:
    return half
  half = half.copy()
  for _ in range(passes):
    corners = [_footprint_corners(centres[i], half[i], float(rotations[i, 2]))
               for i in range(len(centres))]
    # Pairwise, not per box. Taking the nearest neighbour of each box hides
    # the case that actually bites: a box touching one neighbour reports a
    # gap of zero and is left alone, while its one-cell gap to a *different*
    # neighbour stays open. That is what crashed the single-rectangle fit --
    # box 19 touched box 20 and sat 0.4 cells from box 15.
    #
    # Siblings count too. A split emits bounding boxes of disjoint point
    # sets, not abutting pieces, so they leave slivers of their own.
    grown = 0
    for i in range(len(centres)):
      for j in range(i + 1, len(centres)):
        gap = _polygon_gap(corners[i], corners[j])
        # Zero means touching or overlapping, which is already safe.
        if 1e-9 < gap < min_gap:
          half[i, :2] += 0.5 * gap + 1e-3
          half[j, :2] += 0.5 * gap + 1e-3
          grown += 1
          corners[i] = _footprint_corners(
              centres[i], half[i], float(rotations[i, 2])
          )
          corners[j] = _footprint_corners(
              centres[j], half[j], float(rotations[j, 2])
          )
    if not grown:
      break
  else:
    logging.warning(
        'gap closing did not reach a fixed point in %d passes; some alleys '
        'narrower than %.1f m remain, and the ghost cells in them will '
        'interpolate from both buildings', passes, min_gap,
    )
  return half


def report_resolvability(
    boxes: dict[str, np.ndarray], dx: float, dz: float
) -> None:
  """Says whether the grid can actually carry the geometry it was handed.

  Two numbers decide that, and neither is visible in the solidity table.

  A box under three cells wide has no cell that is not on its own surface,
  so the ghost-point interpolation has nothing interior to stand on.
  Measured on the fitted Mandeville scene, where 29 of 70 boxes are under
  three cells: 1 041 ghost cells end up with no fluid at all among the eight
  nodes of their interpolation stencil, against zero for terrain alone or
  for terrain plus one well-resolved block.

  A gap under about three cells between neighbouring *buildings* is the same
  problem seen from the fluid side: the flow between them is carried by a
  couple of cells whose stencils reach into both bodies. Parts of a single
  building abut by construction and are excluded, or the figure would
  measure the splitter rather than the spacing of the houses.

  Both are cheap to check here and expensive to discover at step 300.

  Args:
    boxes: The emitted boxes.
    dx: Horizontal spacing.
    dz: Vertical spacing.
  """
  centres = boxes['box_centres']
  half = boxes['box_half_extents']
  yaw = boxes['box_rotations_deg'][:, 2] if len(centres) else np.zeros(0)
  owner = boxes.get('box_building_id', np.arange(len(centres)))
  if not len(centres):
    print('\nno buildings in this scene (terrain only)')
    return

  width = np.stack(
      [2 * half[:, 0] / dx, 2 * half[:, 1] / dx, 2 * half[:, 2] / dz], -1
  )
  narrow = width[:, :2].min(axis=1)

  print('\nwhat the grid can carry (widths and gaps in CELLS):')
  print('  narrowest horizontal side: min %.1f, median %.1f, max %.1f'
        % (narrow.min(), np.median(narrow), narrow.max()))
  # Total extent, not the visible height: every box is sunk `_BURY_DEPTH_M`
  # below the lowest ground under it so it merges with the terrain.
  print('  total vertical extent, burial included: min %.1f, median %.1f'
        % (width[:, 2].min(), np.median(width[:, 2])))
  for threshold in (2, 3, 4):
    count = int((narrow < threshold).sum())
    if count:
      print('  %2d of %d boxes are under %d cells wide%s'
            % (count, len(centres), threshold,
               '  <-- no interior at all' if threshold == 3 else ''))

  corners = [_footprint_corners(centres[i], half[i], float(yaw[i]))
             for i in range(len(centres))]
  # Every pair, not each box's nearest neighbour: a box that touches one
  # neighbour can still sit one cell from another, and a per-box minimum
  # hides exactly that.
  gaps = [
      _polygon_gap(corners[i], corners[j])
      for i in range(len(centres))
      for j in range(i + 1, len(centres))
  ]
  if not gaps:
    print('  only one box, so no gaps')
    return

  gaps = np.asarray(gaps) / dx
  gaps = gaps[gaps < 6.0]          # distant pairs say nothing
  if not len(gaps):
    print('  every pair is clear by 6+ cells')
    return
  # Zero means touching or overlapping. That is **safe**: the union is a
  # single solid and there is no fluid between them. The dangerous case is a
  # gap that is narrow but not zero -- too thin to carry flow, wide enough to
  # hold ghost cells whose stencils reach into both bodies.
  joined = int((gaps < 1e-6).sum())
  narrow = int(((gaps > 1e-6) & (gaps < 3)).sum())
  clear = int((gaps >= 3).sum())
  print('  close pairs: %d touch or overlap (one solid, fine), %d are clear '
        'by 3 to 6 cells' % (joined, clear))
  if narrow:
    print('  %d PAIRS sit in a gap of under 3 cells that is not closed'
          '  <-- too thin to carry flow, wide enough to break the stencils'
          % narrow)


def pad_with_margin(
    terrain: np.ndarray,
    fuel: np.ndarray,
    boxes: dict[str, np.ndarray],
    crop: tuple[float, float, float, float],
    dx: float,
    margin: float,
    target: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], tuple]:
  """Surrounds the built area with clear fluid.

  A building box that reaches a lateral domain face puts two boundary
  conditions on the same cells -- the face's own, and the immersed boundary's
  no-slip -- and nothing arbitrates between them. Measured on this scene: with
  six of fourteen boxes crossing a face the run is non-finite within ten
  steps, and dropping exactly those six makes it stable. Rather than drop
  buildings, move the faces away from them.

  The terrain is extended at its edge value, which is also what
  `Wui.terrain_profile` does beyond the table, so the two agree. The margin
  carries no vegetation.

  Args:
    terrain: `(nx, ny)` terrain height above the floor.
    fuel: `(nx, ny, nz)` bulk density.
    boxes: The fitted boxes, in the solver frame.
    crop: `(west, south, east, north)` in UTM.
    dx: Horizontal spacing.
    margin: Clear fluid to add on each side, in metres. Rounded up to a
      multiple of `_CELL_MULTIPLE` cells so the padded counts stay
      decomposable.
    target: Exact `(nx, ny)` cell counts to reach, or `None` to use `margin`
      alone. When given, `margin` becomes a floor on what each face must still
      carry rather than the amount added.

  Returns:
    `(terrain, fuel, boxes, crop)`, all padded and shifted.

  Raises:
    ValueError: If `target` cannot be reached while honouring `margin`.
  """
  minimum = int(np.ceil(margin / dx / _CELL_MULTIPLE) * _CELL_MULTIPLE)
  if target is None:
    pads = ((minimum, minimum), (minimum, minimum))
  else:
    pads = (
        _target_padding(terrain.shape[0], target[0], minimum, 'x'),
        _target_padding(terrain.shape[1], target[1], minimum, 'y'),
    )
  if not any(sum(pair) for pair in pads):
    return terrain, fuel, boxes, crop

  terrain = np.pad(terrain, pads, mode='edge')
  fuel = np.pad(fuel, pads + ((0, 0),))
  boxes = dict(boxes)
  boxes['box_centres'] = boxes['box_centres'] + np.array(
      [pads[0][0] * dx, pads[1][0] * dx, 0.0], np.float32
  )
  west, south, east, north = crop
  logging.info(
      'padded with (%d, %d) cells in x and (%d, %d) in y, at least %.0f m of '
      'clear fluid on every face; the mesh is now %d x %d',
      pads[0][0], pads[0][1], pads[1][0], pads[1][1],
      min(p for pair in pads for p in pair) * dx,
      terrain.shape[0], terrain.shape[1],
  )
  return (
      terrain,
      fuel,
      boxes,
      (
          west - pads[0][0] * dx,
          south - pads[1][0] * dx,
          east + pads[0][1] * dx,
          north + pads[1][1] * dx,
      ),
  )


def rotate_half_turn(
    terrain: np.ndarray,
    fuel: np.ndarray,
    boxes: dict[str, np.ndarray],
    dx: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
  """Turns the scene through 180 degrees about the vertical axis.

  Both horizontal axes reverse. Reversing only x would be a reflection rather
  than a rotation: it preserves the footprint outline but mirrors it, so every
  building comes out as its own enantiomer and a wall that faced the wind on
  one side of a ridge faces away from it on the other. A half turn keeps the
  scene congruent to itself and only changes which face the wind arrives at.

  Call after `pad_with_margin`, so the frame being reversed is the final one.
  `origin_utm` is unchanged: it names the south-west corner of the domain,
  which is still the south-west corner, now holding what used to be in the
  north-east.

  Args:
    terrain: `(nx, ny)` terrain height above the floor.
    fuel: `(nx, ny, nz)` bulk density.
    boxes: The fitted boxes, in the solver frame.
    dx: Horizontal spacing.

  Returns:
    `(terrain, fuel, boxes)`, rotated.
  """
  terrain = terrain[::-1, ::-1].copy()
  fuel = fuel[::-1, ::-1, :].copy()

  # The span the solver sees, which is node-inclusive: `Wui` derives
  # `lx = (nx - 1) * dx`, so a centre at `x` maps to `lx - x` and not to
  # `nx * dx - x`. Getting that wrong would shift every building half a cell.
  lx = (terrain.shape[0] - 1) * dx
  ly = (terrain.shape[1] - 1) * dx

  boxes = dict(boxes)
  centres = np.array(boxes['box_centres'], np.float32, copy=True)
  centres[:, 0] = lx - centres[:, 0]
  centres[:, 1] = ly - centres[:, 1]
  boxes['box_centres'] = centres

  # `box_rotations_deg` is left alone. A box is symmetric under a half turn,
  # so its yaw is only defined modulo 180 degrees and `yaw + 180` describes
  # the same body. Half extents are likewise unchanged.
  logging.info(
      'rotated the scene half a turn: the wind now arrives at what was the '
      'eastern face'
  )
  return terrain, fuel, boxes


def write_overlay(
    path: str,
    voxels: Voxels,
    boxes: dict[str, np.ndarray],
    crop: tuple[float, float, float, float],
    span: tuple[float, float] | None = None,
) -> None:
  """Draws the fitted boxes over the building footprint.

  The fitting is the step where a quiet error costs a whole simulation, and
  the numbers in the report do not show a box in the wrong place. Look at it.

  The voxels are an input and are never transformed, while the boxes are
  written in the solver frame, so after a half turn the two disagree unless
  the footprint is turned to match. Without `span` the boxes come out rotated
  against an upright background, which reads as a fitting error and is not
  one.

  Args:
    path: PNG to write.
    voxels: The voxel classification.
    boxes: The fitted boxes.
    crop: `(west, south, east, north)` in UTM.
    span: `(lx, ly)` of the solver frame when the scene was turned half a
      turn, or `None` when it was not. The same reflection the scene got.
  """
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  from matplotlib import patches  # pylint: disable=g-import-not-at-top

  west, south, east, north = crop
  building = np.isin(voxels.class_id, BUILDING_CLASSES)
  x, y = voxels.xy()
  keep = building & (x >= west) & (x < east) & (y >= south) & (y < north)

  figure, axes = plt.subplots(figsize=(11, 11 * (north - south) / (east - west)))
  footprint_x = x[keep] - west
  footprint_y = y[keep] - south
  if span is not None:
    footprint_x = span[0] - footprint_x
    footprint_y = span[1] - footprint_y
  axes.scatter(footprint_x, footprint_y, s=0.2, c='0.6',
               linewidths=0, label='building voxels')
  for centre, half, rotation in zip(boxes['box_centres'],
                                    boxes['box_half_extents'],
                                    boxes['box_rotations_deg']):
    # The anchor is the box's own `(-hx, -hy)` corner carried into the world,
    # so the angle has to be the same `+yaw` that carried it there. Negating
    # it instead swings the rectangle by `2*yaw` about that corner, which
    # draws every box but the two at yaw 0 in the wrong place.
    corner = np.array([-half[0], -half[1]])
    rect = patches.Rectangle(
        _from_frame(corner[None, :], rotation[2])[0] + centre[:2],
        2 * half[0], 2 * half[1], angle=rotation[2],
        fill=False, edgecolor='crimson', linewidth=1.2,
    )
    axes.add_patch(rect)
  axes.set_aspect('equal')
  axes.set_xlabel('east of the domain corner (m)')
  axes.set_ylabel('north of the domain corner (m)')
  axes.set_title(
      'fitted building boxes over the voxel footprint'
      + (' (turned half a turn)' if span is not None else '')
  )
  axes.legend(loc='upper right', markerscale=20)
  figure.tight_layout()
  figure.savefig(path, dpi=140)
  plt.close(figure)
  logging.info('overlay written to %s', path)


def main(_):
  voxels = read_voxels(_VOXELS.value)
  grid = read_grid(_GRID.value)
  datum_offset = report_datum_offset(voxels, grid)

  east_full = grid.west + grid.elevation.shape[1] * grid.dx
  crop = (grid.west, grid.south, east_full, grid.north)
  if _HALF_WIDTH.value is not None:
    if _CENTRE_X.value is None or _CENTRE_Y.value is None:
      raise ValueError('--half_width needs --centre_x and --centre_y')
    half = _HALF_WIDTH.value
    crop = (
        max(grid.west, _CENTRE_X.value - half),
        max(grid.south, _CENTRE_Y.value - half),
        min(east_full, _CENTRE_X.value + half),
        min(grid.north, _CENTRE_Y.value + half),
    )
  dx, dz = _DX.value, _DZ.value
  crop = _snap_crop(crop, dx)
  logging.info(
      'domain %.0f x %.0f m, UTM %.0f..%.0f east, %.0f..%.0f north',
      crop[2] - crop[0], crop[3] - crop[1], crop[0], crop[2], crop[1], crop[3],
  )

  terrain = resample_terrain(grid, crop, dx)
  z_floor = float(terrain.min()) - _BURY_DEPTH_M - dz
  canopy_top = float(terrain.max()) + grid.bulk_density.shape[0] * grid.dz
  z_top = max(canopy_top, float(terrain.max())) + _AIR_ABOVE.value
  nz = int(np.ceil((z_top - z_floor) / dz / _CELL_MULTIPLE) * _CELL_MULTIPLE)

  target = None
  if _TARGET_SHAPE.value is not None:
    if len(_TARGET_SHAPE.value) != 3:
      raise ValueError(
          '--target_shape takes three integers, the cell counts in x, y and '
          f'z; got {_TARGET_SHAPE.value}.'
      )
    target = tuple(int(v) for v in _TARGET_SHAPE.value)
    # The vertical is checked here rather than in `pad_with_margin`, which
    # only touches the horizontal: nothing is ever padded in z, so the target
    # has to be met by raising the column itself. Below the derived `nz` the
    # air above the canopy would be less than `--air_above` asked for.
    if target[2] < nz:
      raise ValueError(
          f'--target_shape asks for {target[2]} cells in z, but '
          f'--air_above={_AIR_ABOVE.value:.0f} m above a canopy top of '
          f'{canopy_top - z_floor:.0f} m needs {nz}. Raise the target or '
          'lower --air_above.'
      )
    if target[2] > nz:
      logging.info(
          'raising the column from %d to %d cells, which puts %.0f m of air '
          'above the canopy top instead of %.0f',
          nz, target[2], target[2] * dz - (canopy_top - z_floor),
          _AIR_ABOVE.value,
      )
    nz = target[2]
  fuel = resample_fuel(grid, terrain, crop, dx, dz, z_floor, nz)

  report: list[dict[str, Any]] = []
  if _BUILDINGS.value == 'fitted':
    boxes, report = extract_buildings(
        voxels, grid, crop, _MAX_BOXES.value, _MIN_AREA.value,
        _MIN_BOX_CELLS.value * dx, _CLOSE_GAPS_CELLS.value * dx, z_floor,
    )
  elif _BUILDINGS.value == 'single':
    position = (
        (_SINGLE_X.value, _SINGLE_Y.value)
        if _SINGLE_X.value is not None and _SINGLE_Y.value is not None
        else None
    )
    # Floor-relative in, floor-relative out, matching what
    # `extract_buildings` returns. The terrain array is still absolute at
    # this point and is converted when the payload is written.
    boxes = single_building(
        terrain - z_floor, dx, dz, _SINGLE_CELLS.value, _SINGLE_YAW.value,
        position,
    )
  else:
    boxes = _no_buildings()
    logging.info('no buildings: terrain-only control')

  terrain, fuel, boxes, crop = pad_with_margin(
      terrain, fuel, boxes, crop, dx, _MARGIN.value,
      target=None if target is None else target[:2],
  )
  if _ROTATE_180.value:
    terrain, fuel, boxes = rotate_half_turn(terrain, fuel, boxes, dx)
  logging.info(
      'target mesh %d x %d x %d at %.1f x %.1f x %.1f m (%.2f M cells)',
      terrain.shape[0], terrain.shape[1], nz, dx, dx, dz,
      terrain.shape[0] * terrain.shape[1] * nz / 1e6,
  )
  logging.info(
      'fuel: %.1f%% of cells non-zero, max %.3f kg/m^3',
      100.0 * (fuel > 0).mean(), float(fuel.max()),
  )

  if report:
    print('\nper-building fit (solidity is the fraction of the boxed area '
          'that is really building):')
    print('  %-5s %5s %8s %9s %9s %9s'
          % ('id', 'boxes', 'yaw', 'roof m', 'area m2', 'solidity'))
    for entry in sorted(report, key=lambda e: e['solidity']):
      flag = '' if entry['solidity'] >= 0.75 else '   <-- check the overlay'
      print('  %-5d %5d %7.1f%s %9.1f %9.0f %9.2f%s'
            % (entry['label'], entry['boxes'], entry['yaw_deg'], '\u00b0',
               entry['roof_height_m'], entry['footprint_m2'],
               entry['solidity'], flag))
    solidities = [e['solidity'] for e in report]
    print('\n%d buildings, %d boxes; solidity median %.2f, worst %.2f'
          % (len(report), len(boxes['box_centres']),
             float(np.median(solidities)), min(solidities)))

  report_resolvability(boxes, dx, dz)

  payload = dict(boxes)
  payload.update({
      'terrain_height': (terrain - z_floor).astype(np.float32),
      'fuel_bulk_density': fuel,
      'origin_utm': np.asarray([crop[0], crop[1], z_floor], np.float64),
      'spacing': np.asarray([dx, dx, dz], np.float64),
      'shape': np.asarray([terrain.shape[0], terrain.shape[1], nz], np.int64),
      'margin_m': np.asarray(
          float(np.ceil(_MARGIN.value / dx / _CELL_MULTIPLE)
                * _CELL_MULTIPLE * dx)
      ),
      'metadata': np.asarray(json.dumps({
          'source_voxels': _VOXELS.value,
          'source_grid': _GRID.value,
          'datum_offset_m': datum_offset,
          'rotated_180': bool(_ROTATE_180.value),
          'target_shape': None if target is None else list(target),
          'note': (
              'All geometry is in the solver frame: metres from the '
              'south-west corner of the domain floor, whose UTM position and '
              'elevation are `origin_utm`. The two sources differ by a '
              'vertical datum; nothing here mixes their absolute elevations.'
          ),
          'buildings': report,
      })),
  })
  np.savez_compressed(_OUTPUT.value, **payload)
  logging.info('scene written to %s', _OUTPUT.value)

  if _OVERLAY.value:
    write_overlay(
        _OVERLAY.value, voxels, boxes, crop,
        span=(
            ((terrain.shape[0] - 1) * dx, (terrain.shape[1] - 1) * dx)
            if _ROTATE_180.value
            else None
        ),
    )


if __name__ == '__main__':
  flags.mark_flags_as_required(['voxels', 'grid', 'output'])
  app.run(main)
