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
_MARGIN = flags.DEFINE_float(
    'margin', 16.0,
    'Clear fluid added around the built area, in metres. A building that '
    'touches a domain face has the face boundary condition and the immersed '
    'boundary contending for the same cells, and the run diverges within ten '
    'steps -- measured. The terrain is extended outwards at its edge value '
    'and the margin carries no fuel.',
)
_MAX_BOXES = flags.DEFINE_integer(
    'max_boxes', 3, 'Most boxes allowed per building.'
)
_MIN_AREA = flags.DEFINE_float(
    'min_footprint_area', 40.0,
    'Footprints smaller than this, in square metres, are discarded as debris.',
)

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
  grid = Grid(
      elevation=np.asarray(store['topography/elevation']).astype(np.float64),
      bulk_density=np.asarray(store['tree/bulkDensity']).astype(np.float32),
      west=float(bounds['west']),
      south=float(bounds['south']),
      north=float(bounds['north']),
      dx=float(attrs['dx']),
      dz=float(attrs['dz']),
  )
  logging.info(
      'grid: elevation %.0f..%.0f m (relief %.0f), fuel %s, %.1f%% non-zero',
      grid.elevation.min(), grid.elevation.max(),
      grid.elevation.max() - grid.elevation.min(),
      grid.bulk_density.shape, 100.0 * (grid.bulk_density > 0).mean(),
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
    min_width: Narrowest box to produce, in metres; use the solver's
      horizontal cell size.
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
  centres, half_extents, rotations = [], [], []
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
  boxes_out = {
      'box_centres': np.asarray(centres, np.float32),
      'box_half_extents': np.asarray(half_extents, np.float32),
      'box_rotations_deg': np.asarray(rotations, np.float32),
  }
  logging.info(
      '%d buildings -> %d boxes', len(report), len(centres)
  )
  return boxes_out, report


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
  xx, yy = np.meshgrid(xs, ys, indexing='ij')
  return grid.elevation_at(xx.ravel(), yy.ravel()).reshape(nx, ny)


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
  return np.where(valid, fuel, 0.0).astype(np.float32)


def pad_with_margin(
    terrain: np.ndarray,
    fuel: np.ndarray,
    boxes: dict[str, np.ndarray],
    crop: tuple[float, float, float, float],
    dx: float,
    margin: float,
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

  Returns:
    `(terrain, fuel, boxes, crop)`, all padded and shifted.
  """
  cells = int(np.ceil(margin / dx / _CELL_MULTIPLE) * _CELL_MULTIPLE)
  if cells == 0:
    return terrain, fuel, boxes, crop
  offset = cells * dx
  terrain = np.pad(terrain, cells, mode='edge')
  fuel = np.pad(fuel, ((cells, cells), (cells, cells), (0, 0)))
  boxes = dict(boxes)
  boxes['box_centres'] = boxes['box_centres'] + np.array(
      [offset, offset, 0.0], np.float32
  )
  west, south, east, north = crop
  logging.info(
      'padded with %d cells (%.0f m) of clear fluid on each side; the mesh '
      'is now %d x %d', cells, offset, terrain.shape[0], terrain.shape[1],
  )
  return (
      terrain,
      fuel,
      boxes,
      (west - offset, south - offset, east + offset, north + offset),
  )


def write_overlay(
    path: str,
    voxels: Voxels,
    boxes: dict[str, np.ndarray],
    crop: tuple[float, float, float, float],
) -> None:
  """Draws the fitted boxes over the building footprint.

  The fitting is the step where a quiet error costs a whole simulation, and
  the numbers in the report do not show a box in the wrong place. Look at it.

  Args:
    path: PNG to write.
    voxels: The voxel classification.
    boxes: The fitted boxes.
    crop: `(west, south, east, north)` in UTM.
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
  axes.scatter(x[keep] - west, y[keep] - south, s=0.2, c='0.6',
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
  axes.set_title('fitted building boxes over the voxel footprint')
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
  fuel = resample_fuel(grid, terrain, crop, dx, dz, z_floor, nz)
  boxes, report = extract_buildings(
      voxels, grid, crop, _MAX_BOXES.value, _MIN_AREA.value, dx, z_floor
  )

  terrain, fuel, boxes, crop = pad_with_margin(
      terrain, fuel, boxes, crop, dx, _MARGIN.value
  )
  logging.info(
      'target mesh %d x %d x %d at %.1f x %.1f x %.1f m (%.2f M cells)',
      terrain.shape[0], terrain.shape[1], nz, dx, dx, dz,
      terrain.shape[0] * terrain.shape[1] * nz / 1e6,
  )
  logging.info(
      'fuel: %.1f%% of cells non-zero, max %.3f kg/m^3',
      100.0 * (fuel > 0).mean(), float(fuel.max()),
  )

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
    write_overlay(_OVERLAY.value, voxels, boxes, crop)


if __name__ == '__main__':
  flags.mark_flags_as_required(['voxels', 'grid', 'output'])
  app.run(main)
