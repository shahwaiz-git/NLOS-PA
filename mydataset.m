%{

%}

clc;
clear;
close all;

load density_map.mat;
load density_map_phantom.mat
load tissue.mat
load tissue_phantom.mat
load sensor_mask.mat
path = 'D:\HISLab\DATASET\StripSkullCT_Simulation\';
Nx = 512;
Ny = 512;
[tissue_xindex,tissue_yindex] = find(tissue_phantom == 3);

for i = 1:50:numel(tissue_xindex)
    %% make a disc
    x = tissue_xindex(i);
    y = tissue_yindex(i);
    
    disc = makeDisc(Nx, Ny, x, y, 5);
    if ~all(tissue_phantom(disc==1)==3)
        continue;
    end
    
    %% simulation and save
    data_name = [num2str(x) '_', num2str(y) '.mat'];
    mixed_path = [path 'mixed_signal\' data_name];
    direct_path = [path 'direct_signal\' data_name];
    target_path = [path 'target\' data_name];
    
    if (exist(mixed_path,'file')) && (exist(direct_path,'file')) && (exist(target_path,'file'))
       continue; 
    end
    
    [mixed_signal, target] = simu_fun(density_map, density_map_phantom, sensor_mask,disc);
    [direct_signal,tmp] = simu_fun(tissue, tissue_phantom, sensor_mask,disc);
    save(mixed_path, 'mixed_signal');
    save(direct_path, 'direct_signal');
    save(target_path, 'target');
    
end


