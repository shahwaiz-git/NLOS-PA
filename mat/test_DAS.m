%{

%}

clc;
clear;

load density_map_256.mat;
load density_map_phantom_256.mat
load tissue_256.mat
load tissue_phantom_256.mat

Nx = 256;
Ny = 256;
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
sensor_mask_idx = sortrows(sensor_mask_idx', 2)';

[tissue_xindex,tissue_yindex] = find(tissue_phantom == 3);

%% make a disc
x = 134;
y = 162;

disc = makeDisc(Nx, Ny, x, y, 5);

[mixed_signal, target] = simu_fun(density_map, density_map_phantom, sensor_mask,disc);
[direct_signal,tmp] = simu_fun(tissue, tissue_phantom, sensor_mask,disc);

das_recon_mix = DAS(mixed_signal, sensor_mask_idx, 1/5e6);
das_recon_dir = DAS(direct_signal, sensor_mask_idx, 1/5e6);

figure;
subplot(1,3,1);imagesc(target);title('target');
subplot(1,3,2);imagesc(das_recon_mix);title('das\_recon\_mix');
subplot(1,3,3);imagesc(das_recon_dir);title('das\_recon\_dir');




