%{

%}

clc;
clear;
close all;
x = 250;
y = 300;
fs = 5e6;
%%
load density_map.mat;
load phantom.mat
load tissue.mat
% load sensor_mask.mat
% load sensor_mask_idx.mat
%% 
Nx = 512;
Ny = 512;
sensor_mask = zeros(Nx, Ny);
sensor_radius = 200;    
sensor_angle = 2*pi;     
sensor_pos = [Nx/2, Ny/2];        
num_sensor_points = 25;
sensor_mask_idx = round(makeCartCircle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle));
for i = 1:num_sensor_points
    ix = sensor_mask_idx(1,i);
    iy = sensor_mask_idx(2,i);
    sensor_mask(ix,iy)=1;
end

%%
[sensor_data_mix, TRM_mix] = simu_fun(x,y,density_map, phantom, sensor_mask);
phantom(phantom==1)=0;
[sensor_data_dir, TRM_dir] = simu_fun(x,y,tissue, phantom, sensor_mask);


DAS_mix = DAS(sensor_data_mix, sensor_mask_idx, 1/fs);
DAS_dir = DAS(sensor_data_dir, sensor_mask_idx, 1/fs);

sensor_data_ref = sensor_data_mix - sensor_data_dir;
DAS_ref = DAS(sensor_data_ref, sensor_mask_idx, 1/fs);

DAS_pas_ref = DAS_ref;
DAS_pas_ref(DAS_pas_ref<0)=0;

DAS_neg_ref = DAS_ref;
DAS_neg_ref(DAS_neg_ref>0)=0;
%% figure
%% 不同情况传感器数据及其DAS重建结果
figure;
subplot(3,2,1);imagesc(sensor_data_mix);title('Mixed signal');
subplot(3,2,2);imagesc(DAS_mix);title('Mixed reconstrcution');
subplot(3,2,3);imagesc(sensor_data_dir);title('Direct signal');
subplot(3,2,4);imagesc(DAS_dir);title('Direct reconstrcution');
subplot(3,2,5);imagesc(sensor_data_ref);title('Reflected signal');
subplot(3,2,6);imagesc(DAS_ref);title('Reflected reconstrcution');
%% 反射信号重建结果各种处理图
figure;
subplot(3,2,1);imagesc(DAS_ref);title('反射信号重建');
subplot(3,2,2);imagesc(abs(DAS_ref));title('绝对值');
subplot(3,2,3);imagesc(DAS_pas_ref);title('正数部分');
subplot(3,2,4);imagesc(DAS_neg_ref);title('负数部分');
subplot(3,2,5);imagesc(DAS_pas_ref.*DAS_pas_ref);title('正数部分^2');
subplot(3,2,6);imagesc(DAS_pas_ref.*DAS_pas_ref.*DAS_pas_ref);title('正数部分^3');
%%
figure;
load p0.mat
subplot(3,2,1);imagesc(p0);title('target');
subplot(3,2,2);imagesc(DAS_mix);title('混合信号重建');
subplot(3,2,3); imagesc(DAS_dir); title('直接信号重建');
subplot(3,2,4); imagesc(DAS_ref); title('反射信号重建');
dir2 = DAS_dir.*DAS_dir;dir2=dir2/max(max(dir2));
ref2 = DAS_ref.*DAS_ref;ref2=ref2/max(max(ref2));
subplot(3,2,5); imagesc(dir2); title('直接信号重建^2');
posref2 = DAS_pas_ref.*DAS_pas_ref;posref2=posref2/max(max(posref2));
subplot(3,2,6); imagesc(dir2+posref2); title('直接重建^2+反射重建(正数部分)^2');

%%
figure;
imagesc(DAS_dir+DAS_ref*2);