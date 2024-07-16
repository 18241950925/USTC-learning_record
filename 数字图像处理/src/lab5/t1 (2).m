clc,clear,close

source = im2double(imread('/home/ubuntu/桌面/matlab/image/flower1.jpg'));
psf = fspecial('motion',30,45);%与图像卷积后逼近相机线性运动的滤波器
motion = imfilter(source,psf,'conv','circular');%卷积，二维循环展开
noisy = imnoise(motion,'gauss',0,0.0001);%产生高斯噪声

subplot(2,4,1); imshow(source); title('flower1'); 
subplot(2,4,2); imshow(motion); title('flower1运动模糊'); 
subplot(2,4,3); imshow(deconvwnr(motion,psf)); title('flower1运动模糊逆滤波');  %在不含噪情况下，Wiener 滤波等效于理想的逆滤波
subplot(2,4,4); imshow(deconvwnr(motion,psf,0.0001)); title('flower1运动模糊维纳滤波'); 
subplot(2,4,6); imshow(noisy); title('flower1运动模糊&高斯噪声'); 
subplot(2,4,7); imshow(deconvwnr(noisy,psf)); title('flower1运动模糊&高斯噪声逆滤波'); 
subplot(2,4,8); imshow(deconvwnr(noisy,psf,0.0001/var(motion(:)))); title('flower1运动模糊&高斯噪声维纳滤波');  %正则化参数的值设置为噪声方差的倒数