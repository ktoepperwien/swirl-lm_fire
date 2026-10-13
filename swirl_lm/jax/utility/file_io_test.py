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
"""Tests for `file_io`, and for the solver importing without TensorFlow."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

from absl.testing import absltest
import numpy as np
import swirl_lm
from swirl_lm.jax.utility import file_io
from swirl_lm.jax.utility import file_pb2


def _repo_root() -> str:
  """The directory holding the top-level `swirl_lm` package."""
  return os.path.dirname(os.path.dirname(os.path.abspath(swirl_lm.__file__)))


class FileIoTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmpdir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

  def _write(self, name: str, contents: str) -> str:
    path = os.path.join(self.tmpdir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
      f.write(contents)
    return path

  def test_parse_csv_file_returns_a_column_per_header(self):
    path = self._write('t.csv', 'a,b\n1,2\n3,4\n')

    out = file_io.parse_csv_file(path)

    self.assertEqual(sorted(out), ['a', 'b'])
    np.testing.assert_allclose(out['a'], [1.0, 3.0])
    np.testing.assert_allclose(out['b'], [2.0, 4.0])
    self.assertEqual(out['a'].dtype, np.float32)

  def test_load_file_returns_the_contents(self):
    path = self._write('x.txt', 'hello\nthere\n')

    self.assertEqual(
        file_io.load_file(file_pb2.File(path=path)), 'hello\nthere\n'
    )

  def test_copy_files_flattens_paths_and_keeps_duplicate_basenames(self):
    sources = [
        file_pb2.File(path=self._write('x/a.txt', 'first')),
        file_pb2.File(path=self._write('y/a.txt', 'second')),
        file_pb2.File(path=self._write('b.txt', 'third')),
    ]
    out_dir = os.path.join(self.tmpdir, 'out')
    os.makedirs(out_dir)

    file_io.copy_files(sources, out_dir)

    self.assertEqual(sorted(os.listdir(out_dir)), ['a.txt', 'a.txt.1', 'b.txt'])
    for name, expected in (
        ('a.txt', 'first'), ('a.txt.1', 'second'), ('b.txt', 'third')
    ):
      with open(os.path.join(out_dir, name)) as f:
        self.assertEqual(f.read(), expected)

  def test_copy_files_visits_a_repeated_path_once(self):
    path = self._write('dup.txt', 'only once')
    out_dir = os.path.join(self.tmpdir, 'out')
    os.makedirs(out_dir)

    file_io.copy_files(
        [file_pb2.File(path=path), file_pb2.File(path=path)], out_dir
    )

    self.assertEqual(os.listdir(out_dir), ['dup.txt'])


class ImportsWithoutTensorFlowTest(absltest.TestCase):
  """The JAX solver must not pull TensorFlow in through a file-IO helper.

  TensorFlow was reached from `file_io`, which sits on the import path of
  everything, so a GPU run loaded it to call `tf.io.gfile`. The check runs in
  a subprocess because an in-process one would pass as soon as anything else
  in the same interpreter had already imported TensorFlow.
  """

  def _import_with_tensorflow_blocked(self, module: str):
    script = textwrap.dedent(f'''
        import sys

        class Blocker:
          """Refuses `tensorflow` wherever it is requested from."""

          def find_spec(self, name, path=None, target=None):
            if name == 'tensorflow' or name.startswith('tensorflow.'):
              raise ImportError(f'tensorflow is blocked, as {{name!r}}')
            return None

        sys.meta_path.insert(0, Blocker())
        import {module}
    ''')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [_repo_root()] + ([env['PYTHONPATH']] if 'PYTHONPATH' in env else [])
    )
    env['JAX_PLATFORMS'] = 'cpu'
    return subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

  def test_file_io_imports(self):
    result = self._import_with_tensorflow_blocked(
        'swirl_lm.jax.utility.file_io'
    )
    self.assertEqual(result.returncode, 0, result.stderr[-4000:])

  def test_the_driver_imports(self):
    result = self._import_with_tensorflow_blocked('swirl_lm.jax.core.driver')
    self.assertEqual(result.returncode, 0, result.stderr[-4000:])


if __name__ == '__main__':
  absltest.main()
