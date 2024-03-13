%%
clc;
clear;
close all;

load density_map.mat
load phantom.mat
%% parameter
x=250;
y=300;
PMLSize = 20;
Nx = 512;
Ny = 512;
dx = 1e-4;
dy = 1e-4;
fs = 5e6;
time = 512 - 1;
input_args = {'PMLInside', false, 'PMLSize', PMLSize, 'Smooth', false, 'PlotPML', false};
%% grid
kgrid = makeGrid(Nx, dx, Ny, dy);
kgrid.t_array = 0:1/fs:time/fs;
%% medium
medium.sound_speed = 1.5*density_map;% [m/s]
medium.density = density_map;
% medium.alpha_coeff = 0.75;      % [dB/(MHz^y cm)]
% medium.alpha_power = 1.5;
% medium.BonA = 6;
%% sensor
sensor_mask = zeros(Nx, Ny);
sensor_mask1 = round([linspace(155,135,30);linspace(170,200,30)]);
sensor_mask2 = round([linspace(385,405,30);linspace(200,260,30)]);
sensor_mask_idx = [sensor_mask1, sensor_mask2];
for i=1:30
    sensor_mask(sensor_mask1(1,i),sensor_mask1(2,i))=1;
    sensor_mask(sensor_mask2(1,i),sensor_mask2(2,i))=1;
end
sensor.mask = sensor_mask;
%% source
disc = makeDisc(Nx, Ny, x, y, 5);
p0 = phantom;
[row,col] = find(disc~=0);
p0(sub2ind(size(p0), row, col))= 5;
source.p0 = p0;

%%
sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});

source.p0 = 0; 
sensor.time_reversal_boundary_data = sensor_data;
TRM_reconstruction = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});
DAS_recon = DAS(sensor_data, sensor_mask_idx, 1/fs);