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
"""A library for the buoyant bubble simulation (JAX port).

The buoyant bubble simulation is performed in a quasi-2D domain. The x and y
dimensions are associated with the horizontal and vertical directions,
respectively. The z dimension is dummy, where periodic boundary condition is
applied.

References:
1. Robert, André. 1993. "Bubble Convection Experiments with a Semi-Implicit
Formulation of the Euler Equations." Journal of the Atmospheric Sciences 50
(13): 1865–73.
2. Bryan, George H., and J. Michael Fritsch. 2002. "A Benchmark Simulation for
Moist Nonhydrostatic Numerical Models." Monthly Weather Review 130 (12):
2917–28.
3. Kurowski, Marcin J., Wojciech W. Grabowski, and Piotr K. Smolarkiewicz. 2014.
"Anelastic and Compressible Simulation of Moist Deep Convection." Journal of the
Atmospheric Sciences 71 (10): 3767–87.
"""

import functools

import jax.numpy as jnp
import numpy as np
from swirl_lm.jax.base import parameters as parameters_lib
from swirl_lm.jax.communication import halo_exchange
from swirl_lm.jax.communication import halo_exchange_utils
from swirl_lm.jax.physics.thermodynamics import manager as thermodynamics_manager
from swirl_lm.jax.physics.thermodynamics import water as water_lib
from swirl_lm.jax.utility import types

ScalarField = types.ScalarField
ScalarFieldMap = types.ScalarFieldMap

# The precomputed gas constant for dry air, in units of J/kg/K.
_R_D = 286.69


class BuoyantBubble:
  """A library of the buoyant bubble simulation (JAX port)."""

  def __init__(
      self,
      config: parameters_lib.SwirlLMParameters,
      q_t_init: float = 0.0,
      theta_init: float = 300.0,
      theta_perturb: float = 2.0,
      x_c: float = 1e4,
      x_r: float = 2e3,
      y_c: float = 2e3,
      y_r: float = 2e3,
      vertical_direction: str = 'y',
      geo_physical: bool = True,
  ):
    """Initializes the simulation setup.

    Args:
      config: Simulation parameters.
      q_t_init: Initial total humidity (kg/kg).
      theta_init: Ambient potential temperature (K).
      theta_perturb: Peak potential temperature perturbation (K).
      x_c: Bubble horizontal center (m).
      x_r: Bubble horizontal radius (m).
      y_c: Bubble vertical center (m).
      y_r: Bubble vertical radius (m).
      vertical_direction: Which dimension is vertical ('x', 'y', or 'z').
      geo_physical: Whether to use geophysical pressure.
    """
    self.config = config

    if self.config.thermodynamics is None:
      raise ValueError('The config does not specify a thermodynamics model.')

    thermo_type = self.config.thermodynamics.WhichOneof('thermodynamics_type')
    if thermo_type == 'water':
      self.water = water_lib.Water(self.config)
    else:
      self.thermodynamics = thermodynamics_manager.ThermodynamicsManager(
          self.config
      )

    self._q_t_init = q_t_init
    self._theta_init = theta_init
    self._theta_p = theta_perturb
    self._x_c = x_c
    self._x_r = x_r
    self._y_c = y_c
    self._y_r = y_r
    self._geo_physical = geo_physical
    self._vertical_direction = vertical_direction

  @property
  def _config_thermodynamics(self):
    assert self.config.thermodynamics is not None
    return self.config.thermodynamics

  def _thermo_bubble_radius(self, xx, yy):
    """Computes the distance from the center of the bubble."""
    return jnp.sqrt(
        ((xx - self._x_c) / self._x_r) ** 2
        + ((yy - self._y_c) / self._y_r) ** 2
    )

  def _thermo_bubble_potential_temperature_init(self, l):
    """Initializes the potential temperature of the thermo bubble."""
    return self._theta_init + jnp.where(
        l < 1.0,
        self._theta_p * jnp.cos(np.pi * l / 2.0) ** 2,
        jnp.zeros_like(l),
    )

  def _get_coordinates(self, xx, yy, zz):
    """Returns (horizontal, vertical) coordinates based on direction."""
    x = yy if self._vertical_direction == 'x' else xx
    vertical_coordinate = {'x': xx, 'y': yy, 'z': zz}
    y = vertical_coordinate[self._vertical_direction]
    return x, y

  def _potential_temperature_init_fn(self, xx, yy, zz):
    """Initializes the perturbed potential temperature field."""
    x, y = self._get_coordinates(xx, yy, zz)
    l = self._thermo_bubble_radius(x, y)
    return self._thermo_bubble_potential_temperature_init(l)

  def _thermal_states_init_fn_water(self, varname, xx, yy, zz):
    """Generates init values for `varname` using the water model."""
    x, y = self._get_coordinates(xx, yy, zz)
    ones = jnp.ones_like(x)
    zeros = jnp.zeros_like(x)

    l = self._thermo_bubble_radius(x, y)
    theta = self._thermo_bubble_potential_temperature_init(l)
    q_t = self._q_t_init * ones

    q_v = q_t
    r_m = _R_D * (1.0 - q_t) + self._config_thermodynamics.water.r_v * q_v
    cp_m = (
        1 - q_t
    ) * self.water.cp_d + q_v * self._config_thermodynamics.water.cp_v
    kappa = r_m / cp_m

    height = y if self._geo_physical else zeros

    # Compute the hydrostatic pressure without the temperature perturbation.
    p = self.water.p_ref(height, {})

    # Compute the temperature based on the potential temperature with
    # perturbation in the bubble, and the hydrostatic pressure.
    exner = (p / self.config.p_thermal) ** kappa
    t = self.water.potential_temperature_to_temperature(
        'theta', theta, q_t, zeros, zeros, height
    )

    rho = p / r_m / t
    theta_li = self.water.potential_temperatures(t, q_t, rho, height)[
        'theta_li'
    ]

    q_l, q_i = self.water.equilibrium_phase_partition(t, rho, q_t)
    e = self.water.internal_energy(t, q_t, q_l, q_i)

    if varname == 'T':
      return t
    elif varname == 'theta_v':
      return r_m / _R_D * t / exner
    elif varname == 'theta_li':
      return theta_li
    elif varname == 'rho':
      return rho
    elif varname == 'p':
      return jnp.zeros_like(y)
    elif varname == 'q_t':
      return q_t
    elif varname == 'e_t':
      return self.water.total_energy(e, zeros, zeros, zeros, height)
    elif varname == 'zz':
      return height
    elif varname == 'xx':
      return l
    else:
      raise ValueError(
          f'{varname} is not a valid option. Available options are: '
          '`T`, `p`, `rho`, `theta_v`, `theta_li`, `q_t`, `e_t`, `zz`, `xx`.'
      )

  def _initialize_states(self, params, logical_coordinates, value_fn):
    """Generates device-local states using `value_fn`.

    Creates a meshgrid for the device's local subdomain (including halos)
    using SYMMETRIC padding, then evaluates `value_fn(xx, yy, zz)`.

    Args:
      params: Simulation parameters.
      logical_coordinates: (cx_i, cy_i, cz_i) device position.
      value_fn: Function mapping (xx, yy, zz) -> ScalarField.

    Returns:
      The device-local field.
    """
    gp = params.grid_params
    hw = gp.halo_width

    # Core points per device in each dimension.
    core_nx = gp.nx - 2 * hw
    core_ny = gp.ny - 2 * hw
    core_nz = gp.nz - 2 * hw

    # Total points per device (core + halos).
    local_nx = core_nx + 2 * hw
    local_ny = core_ny + 2 * hw
    local_nz = core_nz + 2 * hw

    # Grid spacings in xyz order.
    dx, dy, dz = gp.to_xyz_order(gp.grid_spacings)
    cx_i, cy_i, cz_i = logical_coordinates

    # Compute device-local 1D coordinates (SYMMETRIC padding style: halo
    # points mirror interior points).
    x_offset = cx_i * core_nx * dx
    y_offset = cy_i * core_ny * dy
    z_offset = cz_i * core_nz * dz

    x_1d = jnp.arange(local_nx) * dx - hw * dx + x_offset
    y_1d = jnp.arange(local_ny) * dy - hw * dy + y_offset
    z_1d = jnp.arange(local_nz) * dz - hw * dz + z_offset

    # Build meshgrid in data_axis_order.
    grids_xyz = (x_1d, y_1d, z_1d)
    grids_data = gp.to_data_axis_order(*grids_xyz)
    mesh = jnp.meshgrid(*grids_data, indexing='ij')
    # Convert back to xyz order for value_fn.
    xx, yy, zz = gp.to_xyz_order(mesh)

    return value_fn(xx, yy, zz)

  def _initialize_states_physical(self, params, logical_coordinates, value_fn):
    """Like _initialize_states but uses PHYSICAL coordinates in halos.

    Uses global_xyz_with_halos so halo regions have correct physical
    coordinates rather than mirrored or zero-padded values.

    Args:
      params: Simulation parameters.
      logical_coordinates: (cx_i, cy_i, cz_i) device position.
      value_fn: Function mapping (xx, yy, zz) -> ScalarField.

    Returns:
      The device-local field.
    """
    gp = params.grid_params
    hw = gp.halo_width

    core_nx = gp.nx - 2 * hw
    core_ny = gp.ny - 2 * hw
    core_nz = gp.nz - 2 * hw
    local_nx = core_nx + 2 * hw
    local_ny = core_ny + 2 * hw
    local_nz = core_nz + 2 * hw

    cx_i, cy_i, cz_i = logical_coordinates
    # global_xyz_with_halos is a tuple of 3 1D arrays in data_axis_order.
    global_coords = gp.global_xyz_with_halos
    core_ns_data = gp.to_data_axis_order(core_nx, core_ny, core_nz)
    local_ns_data = gp.to_data_axis_order(local_nx, local_ny, local_nz)
    device_ids_data = gp.to_data_axis_order(cx_i, cy_i, cz_i)

    local_1d = []
    for dim in range(3):
      start = device_ids_data[dim] * core_ns_data[dim]
      end = start + local_ns_data[dim]
      local_1d.append(global_coords[dim][start:end])

    mesh = jnp.meshgrid(*local_1d, indexing='ij')
    xx, yy, zz = gp.to_xyz_order(mesh)

    return value_fn(xx, yy, zz)

  def _initial_thermal_states(self, params, logical_coordinates):
    """Initializes the thermodynamics states."""
    init_sym = functools.partial(
        self._initialize_states, params, logical_coordinates
    )
    thermo_type = self._config_thermodynamics.WhichOneof('thermodynamics_type')

    if thermo_type == 'water':
      output = {
          'p': init_sym(
              lambda xx, yy, zz: self._thermal_states_init_fn_water(
                  'p', xx, yy, zz
              )
          ),
          'rho': init_sym(
              lambda xx, yy, zz: self._thermal_states_init_fn_water(
                  'rho', xx, yy, zz
              )
          ),
          'q_t': init_sym(
              lambda xx, yy, zz: self._thermal_states_init_fn_water(
                  'q_t', xx, yy, zz
              )
          ),
          'xx': init_sym(
              lambda xx, yy, zz: self._thermal_states_init_fn_water(
                  'xx', xx, yy, zz
              )
          ),
          'zz': init_sym(
              lambda xx, yy, zz: self._thermal_states_init_fn_water(
                  'zz', xx, yy, zz
              )
          ),
      }

      if 'theta_li' in self.config.transport_scalars_names:
        output['theta_li'] = init_sym(
            lambda xx, yy, zz: self._thermal_states_init_fn_water(
                'theta_li', xx, yy, zz
            )
        )
      else:
        raise ValueError(
            "Missing an energy variable. 'theta_li' must be provided."
        )

      if 'T' in self.config.additional_state_keys:
        output['T'] = init_sym(
            lambda xx, yy, zz: self._thermal_states_init_fn_water(
                'T', xx, yy, zz
            )
        )

    elif thermo_type == 'ideal_gas_law':
      output = {
          'p': init_sym(lambda xx, yy, zz: jnp.zeros_like(xx)),
          'theta': init_sym(self._potential_temperature_init_fn),
      }
      if self._geo_physical:

        def zz_fn(xx, yy, zz):
          vertical_coordinate = {'x': xx, 'y': yy, 'z': zz}
          return vertical_coordinate[self._vertical_direction]

        output['zz'] = init_sym(zz_fn)

      states = {'theta': output['theta']}
      additional_states = {'zz': output['zz']} if self._geo_physical else {}
      output['rho'] = self.thermodynamics.update_thermal_density(
          states, additional_states
      )
    else:
      raise ValueError(
          f'{thermo_type} is not a valid option. Available options are: '
          '`water`, `ideal_gas_law`.'
      )

    return output

  def _initial_sgs_states(self, params, logical_coordinates, helper_states):
    """Initializes helper states for sub-grid scale models."""
    if not self.config.use_sgs:
      return {}

    init_sym = functools.partial(
        self._initialize_states, params, logical_coordinates
    )

    output = {
        'nu_t': init_sym(lambda xx, yy, zz: jnp.zeros_like(xx)),
    }
    sgs_type = self.config.sgs_model.WhichOneof('sgs_model_type')  # pyrefly: ignore[missing-attribute]
    if sgs_type == 'smagorinsky_lilly':
      thermo_type = self._config_thermodynamics.WhichOneof(
          'thermodynamics_type'
      )
      if thermo_type == 'water':
        output['theta_v'] = init_sym(
            lambda xx, yy, zz: self._thermal_states_init_fn_water(
                'theta_v', xx, yy, zz
            )
        )
      elif thermo_type == 'ideal_gas_law':
        output['theta_v'] = helper_states['theta']

    return output

  def initial_states(
      self,
      params: parameters_lib.SwirlLMParameters,
      logical_coordinates: tuple[int, int, int],
  ) -> dict[str, ScalarField]:
    """Initializes the simulation.

    This function conforms to the JAX driver's InitFn signature.

    Args:
      params: Simulation parameters.
      logical_coordinates: (cx_i, cy_i, cz_i) device position.

    Returns:
      Dict of device-local initial states.
    """
    init_sym = functools.partial(
        self._initialize_states, params, logical_coordinates
    )

    output = {
        'u': init_sym(lambda xx, yy, zz: jnp.zeros_like(xx)),
        'v': init_sym(lambda xx, yy, zz: jnp.zeros_like(xx)),
        'w': init_sym(lambda xx, yy, zz: jnp.zeros_like(xx)),
    }

    # Initialize thermodynamical states.
    output.update(self._initial_thermal_states(params, logical_coordinates))

    # Initialize helper variables for sub-grid scale model.
    output.update(
        self._initial_sgs_states(
            params, logical_coordinates, helper_states=output
        )
    )

    return output

  def additional_states_update_fn(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
      step_id: int | None = None,
  ) -> ScalarFieldMap:
    """Updates additional_states each step.

    Ensures `zz` has correct halo values via NEUMANN halo exchange. This is
    critical for correct hydrostatic pressure computation at device boundaries.

    Args:
      states: Essential (prognostic) states.
      additional_states: Additional helper states.
      step_id: Current step ID (unused but required by interface).

    Returns:
      Updated additional_states.
    """
    del step_id
    updated = dict(additional_states)

    # Update theta_v if Smagorinsky-Lilly model is used.
    if 'theta_v' in additional_states:
      thermo_type = self._config_thermodynamics.WhichOneof(
          'thermodynamics_type'
      )
      if thermo_type == 'water':
        temperatures = self.water.update_temperatures(states, additional_states)
        updated['theta_v'] = temperatures['theta_v']
      elif thermo_type == 'ideal_gas_law':
        updated['theta_v'] = states['theta']

    # Update zz halos with NEUMANN BC (gradient = grid spacing in the vertical
    # direction). This ensures correct physical coordinates in halo regions.
    if 'zz' in additional_states:
      updated['zz'] = self._update_halos_vertical(additional_states['zz'])

    # Update T if present and using water thermodynamics.
    if (
        'T' in additional_states
        and self._config_thermodynamics.WhichOneof('thermodynamics_type')
        == 'water'
    ):
      temperatures = self.water.update_temperatures(states, updated)
      updated['T'] = temperatures['T']

    return updated

  def _update_halos_vertical(self, zz: ScalarField) -> ScalarField:
    """Updates halos for the vertical coordinate with NEUMANN BCs.

    Uses NEUMANN(grid_spacing) in the vertical direction so that halo points
    have correct physical coordinate values, and NEUMANN(0) in other
    directions.

    Args:
      zz: The vertical coordinate field.

    Returns:
      The field with corrected halo values.
    """
    gp = self.config.grid_params
    dx, dy, dz = gp.to_xyz_order(gp.grid_spacings)

    if self._vertical_direction == 'x':
      bc_val = dx if self._geo_physical else 0.0
      bc = (
          (
              (halo_exchange_utils.BCType.NEUMANN, bc_val),
              (halo_exchange_utils.BCType.NEUMANN, bc_val),
          ),
          (
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
          ),
          (
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
          ),
      )
    elif self._vertical_direction == 'y':
      bc_val = dy if self._geo_physical else 0.0
      bc = (
          (
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
          ),
          (
              (halo_exchange_utils.BCType.NEUMANN, bc_val),
              (halo_exchange_utils.BCType.NEUMANN, bc_val),
          ),
          (
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
          ),
      )
    else:  # 'z'
      bc_val = dz if self._geo_physical else 0.0
      bc = (
          (
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
          ),
          (
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
              (halo_exchange_utils.BCType.NEUMANN, 0.0),
          ),
          (
              (halo_exchange_utils.BCType.NEUMANN, bc_val),
              (halo_exchange_utils.BCType.NEUMANN, bc_val),
          ),
      )

    return halo_exchange.inplace_halo_exchange(
        zz,
        ('x', 'y', 'z'),
        None,  # mesh (None in single-device context)  # pyrefly: ignore[bad-argument-type]
        gp,
        list(gp.to_xyz_order(self.config.periodic_dims)),  # pyrefly: ignore[bad-argument-type]
        bc,
        halo_width=gp.halo_width,
    )

  def water_thermodynamics_preprocess_fn(
      self,
      states: ScalarFieldMap,
      additional_states: ScalarFieldMap,
  ) -> ScalarFieldMap:
    """Initializes thermodynamic quantities (water model preprocessing).

    This is called once at preprocess_step_id=0 to set thermodynamic
    quantities consistently from the initial potential temperature and
    humidity.

    Args:
      states: Essential states.
      additional_states: Additional states (must contain 'xx' and 'zz').

    Returns:
      Updated states dict.
    """
    l = additional_states['xx']
    yy = self._update_halos_vertical(additional_states['zz'])

    theta = self._thermo_bubble_potential_temperature_init(l)
    q_t = self._q_t_init * jnp.ones_like(l)

    q_v = self._q_t_init
    r_m = (
        _R_D * (1.0 - self._q_t_init)
        + self._config_thermodynamics.water.r_v * q_v
    )

    p = self.water.p_ref(yy, {})

    zeros = jnp.zeros_like(l)
    t = self.water.potential_temperature_to_temperature(
        'theta', theta, q_t, zeros, zeros, yy
    )

    rho = p / r_m / t

    q_l, q_i = self.water.equilibrium_phase_partition(t, rho, q_t)

    states_new = dict(states)
    states_new.update({
        'q_t': q_t,
        'rho': rho,
    })

    if 'theta_li' in self.config.transport_scalars_names:
      states_new['theta_li'] = self.water.temperature_to_potential_temperature(
          'theta_li', t, q_t, q_l, q_i, yy, additional_states
      )

    return states_new
