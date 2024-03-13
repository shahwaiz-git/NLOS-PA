function [image] = DAS(sensor_data, sensor_maskidx, dt)

Nx = 256;
Ny = 256;
d = 1e-3;

vs = 1550;

sensor_num = size(sensor_maskidx, 2);

image = zeros(Nx, Ny);
sensor_mask_idx = sortrows(sensor_maskidx', 2)';

for i = 1:Nx
    for j = 1:Ny
        for c = 1:sensor_num;
            x = sensor_mask_idx(1,c);
            y = sensor_mask_idx(2,c);
            dis = sqrt(((x-i)*d)^2 + ((y-j)*d)^2);
            t = floor(dis/vs/dt) + 1;
            if t<size(sensor_data, 2)
                image(i,j) = image(i,j) + sensor_data(c, t);
            end
        end
    end
end

end