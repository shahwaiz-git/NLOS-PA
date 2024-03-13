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

save(['.\sensor_mask_' num2str(sensor_num) '.mat'],'sensor_mask');
save(['.\sensor_mask_idx_' num2str(sensor_num) '.mat'],'sensor_mask_idx');