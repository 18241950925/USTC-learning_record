clc,clear,close
source = imread("/home/ubuntu/桌面/matlab/image/lena.bmp");
%添加噪声
pepper = imnoise(source,'salt & pepper',0.03);
gaussian = imnoise(source,'gaussian');
random = imnoise(source,'speckle');
%均值滤波
pepper_output = imfilter(pepper,fspecial('average',3));
gaussian_output = imfilter(gaussian,fspecial('average',3));
random_output = imfilter(random,fspecial('average',3));
figure();
subplot(2,4,1); imshow(source); title('原图'); 
subplot(2,4,2); imshow(pepper); title('3%椒盐噪声'); 
subplot(2,4,3); imshow(gaussian); title('高斯噪声'); 
subplot(2,4,4); imshow(random); title('随机噪声'); 
subplot(2,4,6); imshow(pepper_output); title('3%椒盐噪声均值滤波'); 
subplot(2,4,7); imshow(gaussian_output); title('高斯噪声均值滤波'); 
subplot(2,4,8); imshow(random_output); title('随机噪声均值滤波'); 

