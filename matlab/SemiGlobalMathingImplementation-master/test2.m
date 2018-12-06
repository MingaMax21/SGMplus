clc
clear

x_inds = [1; 2; 3];

y_inds = [4; 5; 6];

z_inds = [7; 8; 9];

inds = sub2ind([3 3 3], x_inds, y_inds, z_inds);