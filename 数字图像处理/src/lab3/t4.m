clc,clear,close
source = imread("/home/ubuntu/桌面/matlab/image/lena.bmp");
%添加噪声
pepper = imnoise(source,'salt & pepper',0.03);
gaussian = imnoise(source,'gaussian');
random = imnoise(source,'speckle');

%超限邻域平均法
pepper_output = filter_4( pepper,30 );
gaussian_output = filter_4(gaussian,30);
random_output = filter_4(random, 30);
figure();
subplot(2,4,1); imshow(source); title('原图'); 
subplot(2,4,2); imshow(pepper); title('3%椒盐噪声'); 
subplot(2,4,3); imshow(gaussian); title('高斯噪声'); 
subplot(2,4,4); imshow(random); title('随机噪声'); 
subplot(2,4,6); imshow(pepper_output); title('3%椒盐噪声均值滤波'); 
subplot(2,4,7); imshow(gaussian_output); title('高斯噪声均值滤波'); 
subplot(2,4,8); imshow(random_output); title('随机噪声均值滤波'); 

function [output] = filter_4(input,T)
    output = input;
    [r,l] = size(output);
    for i = 2 : (r - 1)
        for j = 2 : (l - 1)
            temp = input(i - 1 : i + 1, j - 1 : j + 1);
            middle_value = median(temp(:));
            if(abs(double(input(i,j)) - double(middle_value)) > T)
                output(i,j) = middle_value;
            end
        end
    end
end
