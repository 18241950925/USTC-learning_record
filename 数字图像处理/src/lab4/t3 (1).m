clc,clear,close

source1 = imread('/home/ubuntu/桌面/matlab/image/pout.bmp');
source2 = imread('/home/ubuntu/桌面/matlab/image/Girl.bmp');
%添加噪声
source1_pepper = imnoise(source1,'salt & pepper',0.03);
source1_gaussian = imnoise(source1,'gaussian');
source2_pepper = imnoise(source2,'salt & pepper',0.03);
source2_gaussian = imnoise(source2,'gaussian');
D = 40;
figure();
subplot(3,3,1); imshow(source1,[]); title('Pout'); 
subplot(3,3,2); imshow(source1_pepper,[]); title('Pout-椒盐噪声'); 
subplot(3,3,3); imshow(source1_gaussian,[]); title('Pout-高斯噪声'); 
subplot(3,3,4); imshow(ILPF(source1_pepper,D),[]); title('Pout-椒盐噪声理想低通滤波器'); 
subplot(3,3,5); imshow(BLPF(source1_pepper,D,1),[]); title('Pout-椒盐噪声巴特沃斯低通滤波器'); 
subplot(3,3,6); imshow(ELPF(source1_pepper,D,2),[]); title('Pout-椒盐噪声高斯低通滤波器');
subplot(3,3,7); imshow(ILPF(source1_gaussian,D),[]); title('Pout-高斯噪声理想低通滤波器'); 
subplot(3,3,8); imshow(BLPF(source1_gaussian,D,1),[]); title('Pout-高斯噪声巴特沃斯低通滤波器'); 
subplot(3,3,9); imshow(ELPF(source1_gaussian,D,2),[]); title('Pout-高斯噪声高斯低通滤波器');

figure();
subplot(3,3,1); imshow(source2,[]); title('Girl'); 
subplot(3,3,2); imshow(source2_pepper,[]); title('Girl-椒盐噪声'); 
subplot(3,3,3); imshow(source2_gaussian,[]); title('Girl-高斯噪声'); 
subplot(3,3,4); imshow(ILPF(source2_pepper,D),[]); title('Girl-椒盐噪声理想低通滤波器'); 
subplot(3,3,5); imshow(BLPF(source2_pepper,D,1),[]); title('Girl-椒盐噪声巴特沃斯低通滤波器'); 
subplot(3,3,6); imshow(ELPF(source2_pepper,D,1),[]); title('Girl-椒盐噪声高斯低通滤波器');
subplot(3,3,7); imshow(ILPF(source2_gaussian,D),[]); title('Girl-高斯噪声理想低通滤波器'); 
subplot(3,3,8); imshow(BLPF(source2_gaussian,D,1),[]); title('Girl-高斯噪声巴特沃斯低通滤波器'); 
subplot(3,3,9); imshow(ELPF(source2_gaussian,D,1),[]); title('Girl-高斯噪声高斯低通滤波器');

%理想低通滤波器
function output = ILPF(input,D0)
    [r,l] = size(input);
    F = fftshift(fft2(input));
    [U,V] = meshgrid(-l/2:l/2-1,-r/2:r/2-1);
    D = hypot(U,V);
    H = D <= D0 ;
    output = abs(ifft2(ifftshift(F.*H)));
end

%巴特沃斯低通滤波器
function output = BLPF(input,D0,n)
    [r,l] = size(input);
    F = fftshift(fft2(input));
    [U,V] = meshgrid(-l/2:l/2-1,-r/2:r/2-1);
    D = hypot(U,V);
    H = 1./(1+((D./D0).^(2*n)));
    output = abs(ifft2(ifftshift(F.*H)));
end

%高斯低通滤波器
function output = ELPF(input,D0,n)
    [r,l] = size(input);
    F = fftshift(fft2(input));
    [U,V] = meshgrid(-l/2:l/2-1,-r/2:r/2-1);
    D = hypot(U,V);
    H = exp(-(D./D0).^n);
    output = abs(ifft2(ifftshift(F.*H)));
end