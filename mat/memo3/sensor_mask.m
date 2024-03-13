sensor_mask = zeros(512, 512);
sensor_mask1 = round([linspace(155,135,30);linspace(170,200,30)]);
sensor_mask2 = round([linspace(385,405,30);linspace(200,260,30)]);
sensor_mask_idx = [sensor_mask1, sensor_mask2];
for i=1:30
    sensor_mask(sensor_mask1(1,i),sensor_mask1(2,i))=1;
    sensor_mask(sensor_mask2(1,i),sensor_mask2(2,i))=1;
end