function [ sensor_data,p0 ] = simu_fun(density_map,phantom, sensor_mask, disc)
%% grid
PMLSize = 20;
Nx = 512;
Ny = 512;
dx = 1e-4;
dy = 1e-4;
kgrid = makeGrid(Nx, dx, Ny, dy);
fs = 5e6;
time = 512 - 1;
kgrid.t_array = 0:1/fs:time/fs;
%% medium
medium.sound_speed = 1.5*density_map;% [m/s]
medium.density = density_map;
medium.alpha_coeff = 0.75;      % [dB/(MHz^y cm)]
medium.alpha_power = 1.5;
medium.BonA = 6;
%% sensor
sensor.mask = sensor_mask;
%% simu
input_args = {'PMLInside', false, 'PMLSize', PMLSize, 'Smooth', false, 'PlotPML', false};

p0 = phantom;
p0(disc==1)= 5;
source.p0 = p0;
sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});

% source.p0 = 0; 
% sensor.time_reversal_boundary_data = sensor_data;
% TRM_reconstruction = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});
end

