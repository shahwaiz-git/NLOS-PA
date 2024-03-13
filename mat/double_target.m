
clc;
clear;
close all;

load density_map_256.mat;
load density_map_phantom_256.mat
load tissue_256.mat
load tissue_phantom_256.mat
load sensor_mask_50.mat
%% grid
PMLSize = 20;
Nx = 256;
Ny = 256;
dx = 1e-3;
dy = 1e-3;
kgrid = makeGrid(Nx, dx, Ny, dy);
fs =  5e6;
time = 1024-1;
kgrid.t_array = 0:1/fs:time/fs;

%% make a disc
x1 = 130;
y1 = 150;
x2 = 100;
y2 = 170;
disc1 = makeDisc(Nx, Ny, x1, y1, 5);
disc2 = zeros(Nx,Ny);
disc2(x2:x2+10,y2:y2+20) = 1;

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

p0 = density_map_phantom;
p0(disc1==1)= 8;
p0(disc2==1)= 8;
source.p0 = p0;
sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});

