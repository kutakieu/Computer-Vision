img = imread('urban_scene.jpg');
img = imresize(img,[512,512]);
img = rgb2gray(img);

fil1 = [1 ,1,1; 1,1,1; 1,1,1];
fil2 = [ 1,1,1; 0,0,0 ; -1,-1,-1];
fil3 = [ -1,-1, -1; -1 , 8, -1; -1,-1,-1];

fil4 = [1,0,-1;1,0,-1;1,0,-1];



output1 = imfilter(img, fil1/3);
output2 = imfilter(img, fil2);
output3 = imfilter(img, fil3);


figure(4)
subplot(2,2,1)
imshow(output1)
title('filter 1')
subplot(2,2,2)
imshow(output2)
title('filter 2')
subplot(2,2,3)
imshow(output3)
title('filter 3')
% subplot(2,2,4)
% imshow(output4)
% title('filter 4')

figure(1)
FFT1 = fft2(output1);
imagesc(20*log10(abs(fftshift(FFT1))));
IFFT1 = ifft2(FFT1);
outIFFT1 = int8(IFFT1/3);

figure(2)
FFT2 = fft2(output2);
imagesc(20*log10(abs(fftshift(FFT2))));
IFFT2 = ifft2(FFT2);
outIFFT2 = int8(IFFT2/3);

figure(3)
FFT3 = fft2(output3);
imagesc(20*log10(abs(fftshift(FFT3))));
IFFT3 = ifft2(FFT3);
outIFFT3 = int8(IFFT3/3);


subplot(3,3,1)
imshow(output1)
title('filter 1')
subplot(3,3,2)
imagesc(20*log10(abs(fftshift(FFT1))));
title('filter 1')
subplot(3,3,3)
imshow(outIFFT1)
title('filter 1')
subplot(3,3,4)
imshow(output2)
title('filter 2')
subplot(3,3,5)
imagesc(20*log10(abs(fftshift(FFT2))));
title('filter 2')
subplot(3,3,6)
imshow(outIFFT2)
title('filter 2')
subplot(3,3,7)
imshow(output3)
title('filter 3')
subplot(3,3,8)
imagesc(20*log10(abs(fftshift(FFT3))));
title('filter 3')
subplot(3,3,9)
imshow(outIFFT3)
title('filter 3')