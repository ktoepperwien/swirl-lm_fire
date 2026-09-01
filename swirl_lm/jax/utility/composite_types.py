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
"""Commonly used types that include simulation framework object types (JAX).

JAX port of `swirl_lm.utility.composite_types`.
"""


from collections.abc import Callable

import jax
import numpy as np
from swirl_lm.jax.utility import types

ScalarFieldMap = types.ScalarFieldMap

# (replica_id, replicas, states, additional_states) -> updated states
StatesUpdateFn = Callable[
    [jax.Array, np.ndarray, ScalarFieldMap, ScalarFieldMap],
    ScalarFieldMap,
]
# (replica_id, replicas, step_id, states, additional_states) -> updated states
AdditionalStatesUpdateFn = Callable[
    [jax.Array, np.ndarray, int, ScalarFieldMap, ScalarFieldMap],
    ScalarFieldMap,
]
