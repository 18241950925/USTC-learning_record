clc,clear,close

source = imread('/home/ubuntu/桌面/matlab/image/pout.bmp');

D = 15;
n = 1;
a = 5;
b = 2;
subplot(2,4,1); imshow(source,[]); title('Pout'); 
subplot(2,4,2); imshow(histeq(uint8(ILPF(source,D,a,b)))); title('Pout理想高频-直方图'); 
subplot(2,4,3); imshow(histeq(uint8(BLPF(source,D,n,a,b)))); title('Pout巴特沃斯高频-直方图'); 
subplot(2,4,4); imshow(histeq(uint8(ELPF(source,D,n,a,b)))); title('Pout高斯高频-直方图');
subplot(2,4,6); imshow(ILPF(histeq(source),D,a,b),[]); title('Pout直方图-理想高频'); 
subplot(2,4,7); imshow(BLPF(histeq(source),D,n,a,b),[]); title('Pout直方图-巴特沃斯高频'); 
subplot(2,4,8); imshow(ELPF(histeq(source),D,n,a,b),[]); title('Pout直方图-高斯高频');

%理想高频滤波器
function output = ILPF(input,D0,a,b)
    [r,l] = size(input);
    F = fftshift(fft2(input));
    [U,V] = meshgrid(-l/2:l/2-1,-r/2:r/2-1);
    D = hypot(U,V);
    H = D > D0 ;
    H = a * H + b;
    output = abs(ifft2(ifftshift(F.*H)));
end

%巴特沃斯高频滤波器
function output = BLPF(input,D0,n,a,b)
    [r,l] = size(input);
    F = fftshift(fft2(input));
    [U,V] = meshgrid(-l/2:l/2-1,-r/2:r/2-1);
    D = hypot(U,V);
    H = 1./(1+((D0./D).^(2*n)));
    H = a * H + b;
    output = abs(ifft2(ifftshift(F.*H)));
end

%高斯高频滤波器
function output = ELPF(input,D0,n,a,b)
    [r,l] = size(input);
    F = fftshift(fft2(input));
    [U,V] = meshgrid(-l/2:l/2-1,-r/2:r/2-1);
    D = hypot(U,V);
    H = exp((-D0./D).^n);
    H = a * H + b;
    output = abs(ifft2(ifftshift(F.*H)));
end
