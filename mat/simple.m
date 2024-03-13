clearvars; 
clc;
close all;

%% kgrid
PML_size = 20; 
Nx = 512;     
Ny = 512;      
dx = 1e-4;
dy = 1e-4;
kgrid = kWaveGrid(Nx, dx, Ny, dy);

%% medium
skull = zeros(Nx, Ny); 
skull(400:415, 160:320) = 1;

density_map = ones(Nx, Ny) * 1000; 
density_map(skull == 1) = 2500; 
medium.sound_speed = 1.55 * density_map; 
medium.density = density_map;

%% source
p0 = zeros(Nx, Ny);
p0(200, 320) = 1;
source.p0 = p0;

%% sensor
mask = zeros(Nx, Ny); 
sensor_idx = round([120*ones(1,30); 200:229]); 
for i = 1:30
    mask(sensor_idx(1, i), sensor_idx(2, i)) = 1;
end
sensor.mask = mask; 

% mask = zeros(Nx, Ny); 
% sensor_idx = round([linspace(154, 186, 32); linspace(189, 118, 32)]); 
% for i = 1:32
%     mask(sensor_idx(1, i), sensor_idx(2, i)) = 1;
% end
% sensor.mask = mask; 
% 
% sensor_n1 = [-(118 - 189), (186 - 154)]; 
% sensor_n1 = sensor_n1 ./ norm(sensor_n1); 
% sensor.directivity_angle = zeros(Nx, Ny); 
% sensor.directivity_angle(mask == 1) = acos(-sensor_n1(1)); 
% 
% sensor.directivity_pattern = 'pressure';
% sensor.directivity_size = 10 * kgrid.dx; 

%% reconstruction
kgrid.makeTime(medium.sound_speed);
input_args = {'PMLInside', false, 'PMLSize', PML_size, 'Smooth', false, 'PlotPML', false};
sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});

%%  kgrid_recon 
source.p0 = 0; 
sensor.time_reversal_boundary_data = sensor_data;

p0_recon = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});
das_reocn = DAS(sensor_data, sensor_idx, kgrid.dt);

%% overview
figure;
subplot(1,2,1); imagesc(p0_recon); hold on; scatter(320,200);
subplot(1,2,2); imagesc(das_reocn); hold on; scatter(320,200);

