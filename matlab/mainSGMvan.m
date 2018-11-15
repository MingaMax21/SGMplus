% Vanilla SGM
% https://github.com/DecaYale/SGM_matlab/blob/master/SGM.m
% Used to prove SGM concept in Matlab
% 
%% Initialize and calculate cost cube
clc
clear
close all

% Read and interpret config file
s = textscan('calib.txt','%s');

% Read raw images
imgL = imread('im0.png');
imgR = imread('im1.png');

% Convert to grayscalse
imgL = rgb2gray(imgL);
imgR = rgb2gray(imgR);

% Downscale images
imgL = imresize(imgL,0.5);
imgR = imresize(imgR,0.5);


% Normalize  !! ALTERS BEHAVIOR OF MATCHING?!?
% imgL = imgL./max(max(imgL));
% imgR = imgR./max(max(imgR));

%Display preprocessed image
figure(1)
hold on
imshow(imgL);
title('Left prepared image')
colormap gray
colorbar
hold off

% Dimension parameters
H = size(imgL,1);
W = size(imgR,2);

% % Disp levels !! "I" ranges across [0-dispLevel] !!
dispLevels = 15;

% Calculate the raw cost
rawCost = rawCostCalculate(imgL,imgR,dispLevels);

% Find path
costCube = zeros(dispLevels,H,W);

imgL = double(imgL);
imgR = double(imgR);

%% Conduct (timed) SGM over all rows, cols
% Consider single loop if scanning only linewise!
tic
for i=1:H
    for j = 1:W
        
        if (j>1)
            priorCostVec =  costCube(:,i,j-1);
            rawCostVec = rawCost(:,i,j);
            path_intensity_grad = abs( imgL(i,j) - imgL(i,j-1) );
            curCostVec = evaluatePath(priorCostVec,rawCostVec,path_intensity_grad);
            costCube(:,i,j)  =  costCube(:,i,j) + curCostVec;
        end
        
        if (i>1 && j>1)
            priorCostVec =  costCube(:,i-1,j-1);
            rawCostVec = rawCost(:,i,j);
            path_intensity_grad = abs( imgL(i,j)- imgL(i-1,j-1) );
            curCostVec = evaluatePath(priorCostVec,rawCostVec,path_intensity_grad);
            costCube(:,i,j)  =  costCube(:,i,j) + curCostVec;
        end
        
        if (i>1)
            priorCostVec =  costCube(:,i-1,j);
            rawCostVec = rawCost(:,i,j);
            path_intensity_grad = abs( imgL(i,j) - imgL(i-1,j) );
            curCostVec = evaluatePath(priorCostVec,rawCostVec,path_intensity_grad);
            costCube(:,i,j)  =  costCube(:,i,j) + curCostVec;
        end
        
        if (i>1 && j<=W-1)
            priorCostVec =  costCube(:,i-1,j+1);
            rawCostVec = rawCost(:,i,j);
            path_intensity_grad = abs( imgL(i,j) - imgL(i-1,j+1));
            curCostVec = evaluatePath(priorCostVec,rawCostVec,path_intensity_grad);
            costCube(:,i,j)  =  costCube(:,i,j) + curCostVec;
        end
        
        %}
    end
end
toc = toc;

[dispImg, I]= min(costCube);

%% Parse and output images

% !! To convert from the floating-point disparity value d [pixels] in the .pfm file to depth Z [mm] the following equation can be used:
% Z = baseline * f / (d + doffs) 

% Parse new (stereo correllation?) image (Values 0-dispLevels)
d = zeros(H,W);
for i = 1:H
    for j = 1: W
        d(i,j) = I(1,i,j);
    end
end
%Normalize image to 0-255
d = d./ max(max(d))*255;

figure(2)
hold on
imshow(uint8(d));        
title('Correlation Image')
colormap gray
colorbar
hold off

% Parse new disparity image (values in pixels)
di = zeros(H,W);
for i = 1:H
    for j = 1: W
        di(i,j) = dispImg(1,i,j);
    end
end

% Normalize image to 0-255
di = di./ max(max(di))*255;

figure(3)
hold on
imshow(uint8(di));
title('Disparity image')
colormap gray
colorbar
hold off