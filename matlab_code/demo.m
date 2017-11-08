clc;
clear;
im = imread('inputs/7--129.jpg');

I = PencilDrawing(im, 8, 1, 8, 1.0, 1.0);

figure, imshow(I)
