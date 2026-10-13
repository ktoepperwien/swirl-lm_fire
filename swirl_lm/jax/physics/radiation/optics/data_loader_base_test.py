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
"""Tests for staging a remote RRTMGP file where netCDF4 can read it."""

import os
import shutil
import tempfile

from absl.testing import absltest
from swirl_lm.jax.physics.radiation.optics import data_loader_base

# Bytes that are not valid UTF-8, as any real netCDF file holds, so that the
# staged copy is checked as a copy and not just as readable text.
_PAYLOAD = b'CDF\x01\x00\x80\xfe\xff\n\x00binary'


class CreateLocalFileTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmpdir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
    self.staging = os.path.join(self.tmpdir, 'staging')
    self.enter_context(
        absltest.mock.patch.object(
            data_loader_base.DataLoaderBase,
            '_NETCDF_DATA_DIR',
            self.staging,
        )
    )

  def _source(self, name: str = 'rrtmgp.nc') -> str:
    path = os.path.join(self.tmpdir, name)
    with open(path, 'wb') as f:
      f.write(_PAYLOAD)
    return path

  def test_a_binary_file_arrives_byte_for_byte(self):
    local = data_loader_base.DataLoaderBase._create_local_file(self._source())

    self.assertEqual(local, os.path.join(self.staging, 'rrtmgp.nc'))
    with open(local, 'rb') as f:
      self.assertEqual(f.read(), _PAYLOAD)

  def test_the_staging_directory_is_created_on_demand(self):
    self.assertFalse(os.path.exists(self.staging))

    data_loader_base.DataLoaderBase._create_local_file(self._source())

    self.assertTrue(os.path.isdir(self.staging))

  def test_a_file_already_staged_is_not_copied_again(self):
    source = self._source()
    local = data_loader_base.DataLoaderBase._create_local_file(source)

    # Overwrite the source: a second call must leave the staged copy as it is,
    # which is what makes repeated loads cheap.
    with open(source, 'wb') as f:
      f.write(b'different')
    again = data_loader_base.DataLoaderBase._create_local_file(source)

    self.assertEqual(again, local)
    with open(local, 'rb') as f:
      self.assertEqual(f.read(), _PAYLOAD)


if __name__ == '__main__':
  absltest.main()
