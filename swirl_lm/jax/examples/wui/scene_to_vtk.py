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
"""Exports a WUI scene archive for ParaView.

`extract_scene.py` writes a `.npz` that nothing can look at. This turns it
into three files that ParaView opens directly:

  * `<out>.vti` -- the fuel field as cell data on the solver's own mesh, with
    height above the local terrain alongside it so the canopy can be isolated
    without knowing where the ground is;
  * `<out>_terrain.vti` -- the terrain as a 2D height map. Apply **Warp By
    Scalar** to stand it up as a surface;
  * `<out>_boxes.vtu` -- the building boxes as hexahedra, in place and at
    their fitted yaw.

Everything is in the solver frame: metres from the south-west corner of the
domain floor. The archive's `origin_utm` says where that corner is, and is
written into the files as field data rather than applied, so the three line up
with each other rather than with a map.

NumPy only, no VTK bindings, following `combustion_npz_to_vti.py`: this has to
run wherever the archive is, which includes machines where installing VTK is
not worth the trouble.

Example:

    python -m swirl_lm.jax.examples.wui.scene_to_vtk \\
        --scene=.../wui_1m_180deg_rot.npz --output_prefix=.../wui_1m

A full 512 x 512 x 256 fuel field is 268 MB per array, so `--stride` thins the
3D grid; the terrain and the boxes are always written whole.

In ParaView: open the `.vti`, **Threshold** on `rho_f` above zero to drop the
empty 99.8% of cells, and colour by `rho_f`. Load the `.vtu` alongside for the
buildings and the warped terrain for the ground.
"""

from collections.abc import Sequence
import json
import os
import struct

from absl import app
from absl import flags
from absl import logging
import numpy as np


_SCENE = flags.DEFINE_string('scene', None, 'Scene `.npz` to read.')
# `output_prefix` rather than `output`: `extract_scene` already owns the
# latter, and absl refuses a duplicate as soon as anything imports both --
# which the tests do.
_OUTPUT = flags.DEFINE_string(
    'output_prefix', None,
    'Path prefix; `.vti`, `_terrain.vti` and `_boxes.vtu` are appended.',
)
_STRIDE = flags.DEFINE_integer(
    'stride', 1,
    'Take every nth cell of the 3D field on each axis. The fuel occupies a '
    'fraction of a percent of cells on this scene, so a stride above one '
    'will drop most of it -- use it to find your way around, not to look at '
    'the canopy.',
)
_FIELDS = flags.DEFINE_list(
    'fields', ['rho_f', 'hag'],
    'Which 3D arrays to write: `rho_f`, the fuel bulk density, and `hag`, '
    'height above the local terrain.',
)

# VTK's own cell type number for a trilinear hexahedron.
_VTK_HEXAHEDRON = 12

_NUMPY_TO_VTK = {
    np.dtype(np.float32): 'Float32',
    np.dtype(np.float64): 'Float64',
    np.dtype(np.int32): 'Int32',
    np.dtype(np.int64): 'Int64',
    np.dtype(np.uint8): 'UInt8',
}


def _appended(arrays: Sequence[np.ndarray]) -> tuple[list[int], bytes]:
  """Lays arrays out as a VTK appended-data block.

  Each array is preceded by its length in bytes as a `UInt64`, which is what
  `header_type="UInt64"` in the file header promises.

  Args:
    arrays: The arrays, already in VTK's x-fastest order.

  Returns:
    `(offsets, payload)`, the byte offset of each array's header and the
    concatenated block.
  """
  offsets, chunks, position = [], [], 0
  for array in arrays:
    data = np.ascontiguousarray(array).tobytes()
    offsets.append(position)
    chunks.append(struct.pack('<Q', len(data)))
    chunks.append(data)
    position += 8 + len(data)
  return offsets, b''.join(chunks)


def _to_vtk_order(array: np.ndarray) -> np.ndarray:
  """Reorders an `(nx, ny, nz)` array so x varies fastest, as VTK expects.

  The archive is indexed x first, which in C order makes z the fastest axis --
  exactly the opposite. Writing it without this transpose produces a file that
  opens, has the right size, and shows the domain transposed, which is the
  kind of wrong that survives a glance.

  Args:
    array: An `(nx, ny, nz)` array.

  Returns:
    The same values, flattened x-fastest.
  """
  return np.ascontiguousarray(array.transpose(2, 1, 0)).ravel()


def _write_image(
    path: str,
    cell_arrays: dict[str, np.ndarray],
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    field: dict[str, str],
) -> None:
  """Writes cell-centred arrays as a VTK ImageData file.

  Args:
    path: Where to write.
    cell_arrays: Name to `(nx, ny, nz)` array.
    shape: `(nx, ny, nz)` cell counts.
    spacing: Cell size on each axis.
    field: Strings to carry along as field data, such as the UTM origin.
  """
  nx, ny, nz = shape
  names = list(cell_arrays)
  flat = [_to_vtk_order(cell_arrays[name]) for name in names]
  offsets, payload = _appended(flat)

  # The extent counts *points*, so `nx` cells spans `0..nx`.
  header = [
      '<?xml version="1.0"?>',
      '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian"'
      ' header_type="UInt64">',
      f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="0 0 0"'
      f' Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">',
      '    <FieldData>',
  ]
  for key, value in field.items():
    header.append(
        f'      <DataArray type="String" Name="{key}" NumberOfTuples="1"'
        ' format="ascii">'
        + ' '.join(str(ord(c)) for c in value) + ' 0</DataArray>'
    )
  header += [
      '    </FieldData>',
      f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">',
      f'      <CellData Scalars="{names[0]}">',
  ]
  for name, array, offset in zip(names, flat, offsets):
    header.append(
        f'        <DataArray type="{_NUMPY_TO_VTK[array.dtype]}"'
        f' Name="{name}" NumberOfComponents="1" format="appended"'
        f' offset="{offset}"/>'
    )
  header += [
      '      </CellData>',
      '    </Piece>',
      '  </ImageData>',
      '  <AppendedData encoding="raw">',
      '   _',
  ]
  with open(path, 'wb') as out:
    out.write('\n'.join(header).encode('ascii'))
    out.write(payload)
    out.write(b'\n  </AppendedData>\n</VTKFile>\n')
  logging.info(
      'wrote %s: %d x %d x %d cells, %s, %.0f MB',
      path, nx, ny, nz, ', '.join(names), os.path.getsize(path) / 1e6,
  )


def box_corners(
    centre: np.ndarray, half: np.ndarray, yaw_deg: float
) -> np.ndarray:
  """The eight corners of one fitted box, in the solver frame.

  The yaw convention has to be the one `signed_distance.batched_boxes` reads,
  which carries the box's own axes into the world by `+yaw`. Negating it draws
  every box but those at yaw zero somewhere else -- the mistake the fitting
  overlay made for an afternoon.

  Args:
    centre: `(3,)` box centre.
    half: `(3,)` half extents.
    yaw_deg: Rotation about the vertical axis, in degrees.

  Returns:
    An `(8, 3)` array of corners, in VTK's hexahedron order.
  """
  # VTK orders a hexahedron as the four base corners anticlockwise, then the
  # four above them in the same order.
  signs = np.array([
      [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
      [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
  ], np.float64)
  local = signs * half[None, :]
  angle = np.radians(yaw_deg)
  cos, sin = np.cos(angle), np.sin(angle)
  rotated = np.stack([
      cos * local[:, 0] - sin * local[:, 1],
      sin * local[:, 0] + cos * local[:, 1],
      local[:, 2],
  ], axis=1)
  return rotated + centre[None, :]


def _write_boxes(path: str, scene, field: dict[str, str]) -> None:
  """Writes the building boxes as an unstructured grid of hexahedra."""
  centres = np.asarray(scene['box_centres'], np.float64)
  halves = np.asarray(scene['box_half_extents'], np.float64)
  yaws = np.asarray(scene['box_rotations_deg'], np.float64)
  if not len(centres):
    logging.warning('no boxes in this scene; skipping %s', path)
    return

  points = np.concatenate([
      box_corners(centres[k], halves[k], yaws[k, 2])
      for k in range(len(centres))
  ]).astype(np.float32)
  count = len(centres)
  connectivity = np.arange(8 * count, dtype=np.int64)
  offsets = np.arange(1, count + 1, dtype=np.int64) * 8
  types = np.full(count, _VTK_HEXAHEDRON, np.uint8)
  identifiers = (
      np.asarray(scene['box_building_id'], np.int32)
      if 'box_building_id' in scene.files
      else np.arange(count, dtype=np.int32)
  )

  arrays = [points.ravel(), connectivity, offsets, types, identifiers]
  positions, payload = _appended(arrays)
  header = [
      '<?xml version="1.0"?>',
      '<VTKFile type="UnstructuredGrid" version="1.0"'
      ' byte_order="LittleEndian" header_type="UInt64">',
      '  <UnstructuredGrid>',
      '    <FieldData>',
  ]
  for key, value in field.items():
    header.append(
        f'      <DataArray type="String" Name="{key}" NumberOfTuples="1"'
        ' format="ascii">'
        + ' '.join(str(ord(c)) for c in value) + ' 0</DataArray>'
    )
  header += [
      '    </FieldData>',
      f'    <Piece NumberOfPoints="{8 * count}"'
      f' NumberOfCells="{count}">',
      '      <Points>',
      '        <DataArray type="Float32" NumberOfComponents="3"'
      f' format="appended" offset="{positions[0]}"/>',
      '      </Points>',
      '      <Cells>',
      '        <DataArray type="Int64" Name="connectivity"'
      f' format="appended" offset="{positions[1]}"/>',
      '        <DataArray type="Int64" Name="offsets"'
      f' format="appended" offset="{positions[2]}"/>',
      '        <DataArray type="UInt8" Name="types"'
      f' format="appended" offset="{positions[3]}"/>',
      '      </Cells>',
      '      <CellData Scalars="building_id">',
      '        <DataArray type="Int32" Name="building_id"'
      f' NumberOfComponents="1" format="appended" offset="{positions[4]}"/>',
      '      </CellData>',
      '    </Piece>',
      '  </UnstructuredGrid>',
      '  <AppendedData encoding="raw">',
      '   _',
  ]
  with open(path, 'wb') as out:
    out.write('\n'.join(header).encode('ascii'))
    out.write(payload)
    out.write(b'\n  </AppendedData>\n</VTKFile>\n')
  logging.info('wrote %s: %d boxes', path, count)


def main(argv: Sequence[str]) -> None:
  del argv
  if _SCENE.value is None or _OUTPUT.value is None:
    raise ValueError('--scene and --output_prefix are both required.')

  scene = np.load(_SCENE.value, allow_pickle=False)
  terrain = np.asarray(scene['terrain_height'], np.float32)
  fuel = np.asarray(scene['fuel_bulk_density'], np.float32)
  spacing = np.asarray(scene['spacing'], np.float64)
  origin = np.asarray(scene['origin_utm'], np.float64)
  metadata = json.loads(str(scene['metadata']))

  provenance = {
      'origin_utm': ' '.join(f'{v:.3f}' for v in origin),
      'rotated_180': str(bool(metadata.get('rotated_180', False))),
      'source_grid': str(metadata.get('source_grid', '')),
  }
  logging.info(
      'scene %s: %s at %s m, rotated_180=%s',
      _SCENE.value, tuple(np.asarray(scene['shape'])), tuple(spacing),
      provenance['rotated_180'],
  )

  stride = max(1, _STRIDE.value)
  sliced = (slice(None, None, stride),) * 3
  available = {}
  if 'rho_f' in _FIELDS.value:
    available['rho_f'] = fuel[sliced]
  if 'hag' in _FIELDS.value:
    # Height above the local terrain, which is what separates canopy from air
    # without having to know the ground. Cell centres, as the fuel is.
    nz = fuel.shape[2]
    centres = (np.arange(nz, dtype=np.float32) + 0.5) * spacing[2]
    available['hag'] = (centres[None, None, :] - terrain[:, :, None])[sliced]
  if not available:
    raise ValueError(f'--fields selected nothing from {_FIELDS.value}.')

  shape = next(iter(available.values())).shape
  _write_image(
      f'{_OUTPUT.value}.vti',
      available,
      shape,
      tuple(float(s) * stride for s in spacing),
      provenance,
  )

  # The terrain as its own 2D map, one cell thick, so Warp By Scalar stands it
  # up without having to carry the whole column.
  _write_image(
      f'{_OUTPUT.value}_terrain.vti',
      {'terrain_height': terrain[:, :, None]},
      (terrain.shape[0], terrain.shape[1], 1),
      (float(spacing[0]), float(spacing[1]), float(spacing[2])),
      provenance,
  )

  _write_boxes(f'{_OUTPUT.value}_boxes.vtu', scene, provenance)


if __name__ == '__main__':
  app.run(main)
