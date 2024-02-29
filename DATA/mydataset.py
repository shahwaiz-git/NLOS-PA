import mat73
import scipy
import numpy as np
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.conversion import cart2grid
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.mapgen import make_cart_circle, make_disc
from kwave.utils.signals import reorder_binary_sensor_data

phantom_path = '../phantom.mat'
density_map_path = '../density_map.mat'
phantom = mat73.loadmat(phantom_path)['phantom']
density_map = mat73.loadmat(density_map_path)['density_map']



# grid
PMLSize = Vector([20, 20])
grid_dsize = Vector([1e-4, 1e-4])
grid_pixel = Vector([512, 512])
kgrid = kWaveGrid(grid_pixel, grid_dsize)

# medium
medium = kWaveMedium(
    sound_speed=1.5*density_map,
    density=density_map,
    alpha_coeff=0.75,
    alpha_power=1.5,
    BonA=6
)
kgrid.makeTime(medium.sound_speed)

# sensor
sensor = kSensor()
sensor_mask = np.zeros(grid_pixel)
sensor_radius = 200
sensor_angle = 2*np.pi
sensor_pos = grid_dsize / 2
sensor_points = 100
sensor_mask_idx = make_cart_circle(
    radius=sensor_radius,
    num_points=sensor_points,
    arc_angle=sensor_angle,
).astype(np.int16)
for x, y in sensor_mask_idx.T:
    sensor_mask[x, y] = 1
sensor.mask = sensor_mask

# source
source = kSource()
x = 250
y = 300
source_radius = 5
disc = make_disc(
    grid_size=grid_pixel,
    center=Vector([x, y]),
    radius=source_radius
)

source.p0 = phantom + disc

# simulation
simulation_options = SimulationOptions(
    pml_size=PMLSize,
    pml_inside=False,
    save_to_disk=True
)
execution_options = SimulationExecutionOptions(is_gpu_simulation=True)

sensor_data = kspace_first_order_2d_gpu(
    kgrid=kgrid,
    source=source,
    sensor=sensor,
    medium=medium,
    simulation_options=simulation_options,
    execution_options=execution_options)