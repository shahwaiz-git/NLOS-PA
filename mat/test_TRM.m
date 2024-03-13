clc;
clear;

load density_map_256.mat
load density_map_phantom_256.mat

%% grid
PMLSize = 20;
Nx = 256;
Ny = 256;
dx = 1e-4;
dy = 1e-4;
kgrid = makeGrid(Nx, dx, Ny, dy);

%% medium
medium.sound_speed = 1.5*density_map;% [m/s]
medium.density = density_map;
medium.alpha_coeff = 0.75;      % [dB/(MHz^y cm)]
medium.alpha_power = 1.5;
medium.BonA = 6;

%% sensor           
sensor_mask = zeros(Nx, Ny);
sensor_radius = 120;    
sensor_angle = 2*pi;     
sensor_pos = [Nx/2, Ny/2];        
sensor_num = 50;
sensor_mask_idx = round(makeCartCircle(sensor_radius, sensor_num, sensor_pos, sensor_angle));
for i = 1:sensor_num
    ix = sensor_mask_idx(1,i);
    iy = sensor_mask_idx(2,i);
    sensor_mask(ix,iy)=1;
end
sensor.mask = sensor_mask;

%% simu
x = 134;
y = 162;
disc = makeDisc(Nx, Ny, x, y, 5);

p0 = density_map_phantom;
p0(disc==1)= 8;
source.p0 = p0;

input_args = {'PMLInside', false, 'PMLSize', PMLSize, 'Smooth', false, 'PlotPML', false};

sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});
DAS_reconstruction = DAS(sensor_data, sensor_mask_idx, kgrid.dt);

source.p0 = 0; 
sensor.time_reversal_boundary_data = sensor_data;

TRM_reconstruction = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});


figure;
subplot(1,3,1);imagesc(p0);
subplot(1,3,2);imagesc(TRM_reconstruction);
subplot(1,3,3);imagesc(DAS_reconstruction);
