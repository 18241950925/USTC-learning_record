%clear
clc,clear,close all

%1、灰度的线性变换

input_image = imread("/home/ubuntu/桌面/matlab/image/lena.bmp");
fA = 1.5; % 斜率
fB = 30; % 截距

output_image = fA * input_image + fB;
subplot(1,2,1)
imshow(input_image);
subplot(1,2,2);
imshow(output_image);

%2、灰度拉伸

input_image = imread("/home/ubuntu/桌面/matlab/image/lena.bmp");
x1 = 100;
y1 = 50; 
x2 = 200; 
y2 = 200; 
% 应用灰度拉伸
output_image = input_image; 
output_image(input_image < x1) = y1 * input_image(input_image < x1) / x1;
output_image((input_image >= x1) & (input_image <= x2)) = (y2 - y1) * (input_image((input_image >= x1) & (input_image <= x2)) - x1) / (x2 - x1) + y1;
output_image(input_image > x2) = (255 - y2) * (input_image(input_image > x2) - x2) / (255 - x2) + y2;
figure;
imshow(output_image);

%3、灰度直方图

input_image = imread("/home/ubuntu/桌面/matlab/image/lena.bmp");
figure;
histogram(input_image);
lower_x = 50;
upper_x = 200; 
lower_y = 0;
upper_y = 3000;
axis([lower_x ,upper_x,lower_y ,upper_y]);


%4、直方图均衡

input_image = imread("/home/ubuntu/桌面/matlab/image/pout.bmp");

figure;
subplot(321);
imshow(input_image);
subplot(3,2,2);
histogram(input_image);
% 应用直方图均衡
output_image = histeq(input_image);
subplot(3,2,3)
imshow(output_image);
title('直方图均衡')
subplot(324);
histogram(output_image);
% 原始图像 pout.bmp 进行直方图规定化处理，将直方图规定化为高斯分布
%均值127 方差30的高斯分布
output2 = histeq(input_image,normpdf((0:1:255),127,30));

% 显示规定化后的图像及其直方图
subplot(3,2,5);
imshow(output2);
title('高斯分布图像');
subplot(3,2,6);
histogram(output2);
