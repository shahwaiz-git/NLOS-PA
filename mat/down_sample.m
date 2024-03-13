load density_map.mat
load density_map_phantom.mat
load tissue.mat
load tissue_phantom.mat

density_map = density_map(1:2:end,1:2:end);
density_map_phantom = density_map_phantom(1:2:end,1:2:end);
tissue = tissue(1:2:end,1:2:end);
tissue_phantom = tissue_phantom(1:2:end,1:2:end);


save('density_map_256','density_map');
save('density_map_phantom_256','density_map_phantom');
save('tissue_256','tissue');
save('tissue_phantom_256','tissue_phantom');