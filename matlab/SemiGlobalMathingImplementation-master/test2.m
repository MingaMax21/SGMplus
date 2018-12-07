clc
clear

mat1 = [[1,2,3];[4,5,6];[7,8,9]];
mat2 = [[10,11,12];[13,14,15];[16,17,18]];
mat3 = [[19,20,21];[22,23,24];[25,26,27]];

mat = cat(3,mat1,mat2,mat3);
first = mat(1,1,1);
last  = mat(3,3,3);

[x, dMap] = min(mat, [], 3);

dMap2 = min(mat, [], 3);