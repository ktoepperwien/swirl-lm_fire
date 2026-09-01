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

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Commonly used types in the LPT module (JAX port).

JAX port of `swirl_lm.physics.lpt.lpt_types`.
"""


import jax
import jax.numpy as jnp
import numpy as np

LPT_INT = jnp.int32
LPT_FLOAT = jnp.float32

LPT_NP_FLOAT = np.float32
LPT_NP_INT = np.int32

# LPT `ScalarFieldMap` dictionary keys.
LPT_INTS_KEY = 'lpt_ints'
LPT_FLOATS_KEY = 'lpt_floats'
LPT_COUNTER_KEY = 'lpt_new_particle_counter'

LptFieldInts = jax.Array
LptFieldFloats = jax.Array
LptCoord = tuple[float | jax.Array, float | jax.Array, float | jax.Array]

# LPT Field int column indices.

# The status of the particle, 1=active, 0=inactive.
COL_STATUS = 0
# The globally unique ID of the particle.
COL_ID = 1

# LPT Field float column indices.

# The z coordinate, in a uniform mesh.
COL_X0 = 0
# The x coordinate, in a uniform mesh.
COL_X1 = 1
# The y coordinate, in a uniform mesh.
COL_X2 = 2
# The particle velocity along the z coordinate, in units of m/s.
COL_V0 = 3
# The particle velocity along the x coordinate, in units of m/s.
COL_V1 = 4
# The particle velocity along the y coordinate, in units of m/s.
COL_V2 = 5
# The mass of the particle, in units of kg.
COL_MASS = 6
