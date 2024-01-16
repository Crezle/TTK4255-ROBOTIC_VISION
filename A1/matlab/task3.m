clear;
filename       = '../data/grid.jpg';
edge_threshold = 1; % todo: choose an appropriate value
blur_sigma     = 1; % todo: choose an appropriate value

I_rgb       = im2double(imread(filename));
I_gray      = rgb_to_gray(I_rgb);
I_blur      = gaussian(I_gray, blur_sigma);
[Ix,Iy,Im]  = central_difference(I_blur);
[x,y,theta] = extract_edges(Ix, Iy, Im, edge_threshold);

figure(1);
set(gcf,'Position',[100 100 1000 300])
subplot(151); imshow(I_blur);            xlim([300, 500]); title('Blurred input');
subplot(152); imshow(Ix, [-0.05, 0.05]); xlim([300, 500]); title('Gradient in x');
subplot(153); imshow(Iy, [-0.05, 0.05]); xlim([300, 500]); title('Gradient in y');
subplot(154); imshow(Im, [ 0.00, 0.05]); xlim([300, 500]); title('Gradient magnitude');
subplot(155);
scatter(x, y, 1, theta);
colormap(gca, 'hsv');
box on; axis image;
set(gca, 'YDir', 'reverse');
xlim([300, 500]);
ylim([0, size(I_rgb,1)]);
title('Extracted edge points');
c = colorbar('southoutside');
c.Label.String = 'Edge angle (radians)';
