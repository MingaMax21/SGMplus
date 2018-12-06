clc
clear

disparity_range = [-16 16];
im_reference  = rgb2gray(imread('image_left.png'));
im_support  = rgb2gray(imread('image_right.png'));

% output image:
C = zeros(size(im_support,1), size(im_reference,2), disparity_range(2) - disparity_range(1) + 1, 'double');

% allocate im_diff. im_diff is in left coords:
im_diff = zeros(size(im_reference, 1), size(im_reference,2), 'double');
Lr = zeros([size(C),8], 'double');
for disp = 5
    
    % calc borders for images
    init = [1, size(im_reference,2)];
    borders_left = [1, size(im_reference,2)] - disp*  [(disp < 0), disp > 0];      
    borders_right = borders_left + [disp, disp];
    
    % im_diff = (im_left - im_reference).^2, inside borders
    im_diff(:, borders_left(1) : borders_left(2)) = ...
    im_support(:, borders_left(1) : borders_left(2)) - ...
    im_reference(:, borders_right(1) : borders_right(2));
    im_diff = abs(im_diff);
    
    im_diff(:, 1:borders_left(1)) = im_diff(:, borders_left(1)) * ...
    ones(1, borders_left(1));
    im_diff(:, borders_left(2):end) = im_diff(:,borders_left(2)) * ...
    ones(1, size(im_diff,2) - borders_left(2) + 1);

     % for summing, do convolution with ones:
    filt = ones(5);
    C(:, :, disp - disparity_range(1) + 1) = ...
        conv2(im_diff, filt, 'same');


end