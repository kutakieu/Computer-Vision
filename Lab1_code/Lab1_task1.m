img = imread('face_01_U5934839.jpg');
img = imrotate(img,-90,'bilinear','loose');
imgGrey = rgb2gray(img);
imgGreyResize = imresize(imgGrey, [512,512]);
imgGreyResizeNoiseG = imnoise(imgGreyResize,'gaussian',0,10/255);
imgGreyResizeNoiseSP = imnoise(imgGreyResize,'salt & pepper',0.1);


figure(1)
subplot(2,2,1);
imshow(imgGreyResize);
title('Original Image');
subplot(2,2,2);
imshow(imgGreyResizeNoiseG);
title('Gaussian noised');
subplot(2,2,3);
imshow(imgGreyResizeNoiseSP);
title('Salt and Pepper noised');


output = imgGreyResize;
my_9x9_gausskernel = fspecial('gaussian',[9 9],0.1);
%imgGreyResizeNoiseG_f1 = my_Gauss_filter(imgGreyResizeNoiseG,my_9x9_gausskernel,output);
imgGreyResizeNoiseG_f1 = my_filter(imgGreyResizeNoiseG,my_9x9_gausskernel );
%imgGreyResizeNoiseG_f = imfilter(imgGreyResizeNoiseG,my_9x9_gausskernel);
SNR = 20*log ( norm(single(imgGreyResize),'fro') /norm(single(imgGreyResize - imgGreyResizeNoiseG_f), 'fro' )) ;
my_9x9_gausskernel = fspecial('gaussian',[9 9],1);
imgGreyResizeNoiseG_f2 = my_Gauss_filter(imgGreyResizeNoiseG,my_9x9_gausskernel,output);
%imgGreyResizeNoiseG_f = imfilter(imgGreyResizeNoiseG,my_9x9_gausskernel);
%SNR = 20*log ( norm(single(imgGreyResize),'fro') /norm(single(imgGreyResize - imgGreyResizeNoiseG_f), 'fro' )) ;
my_9x9_gausskernel = fspecial('gaussian',[9 9],10);
imgGreyResizeNoiseG_f3 = my_Gauss_filter(imgGreyResizeNoiseG,my_9x9_gausskernel,output);
%imgGreyResizeNoiseG_f = imfilter(imgGreyResizeNoiseG,my_9x9_gausskernel);
%SNR = 20*log ( norm(single(imgGreyResize),'fro') /norm(single(imgGreyResize - imgGreyResizeNoiseG_f), 'fro' )) ;

subplot(131);
imshow(imgGreyResizeNoiseG_f1);
title('Standard deviation 0.1');
subplot(132);
imshow(imgGreyResizeNoiseG_f2);
title('Standard deviation 1');
subplot(133);
imshow(imgGreyResizeNoiseG_f3);
title('Standard deviation 10');


imgGreyResizeNoiseSP_f = MyMedianFilter(imgGreyResizeNoiseSP);
% %MyMedianFilter = fspecial();
sobel_filter = [-1,0,1;-2,0,2;-1,0,1];
sobel_filter_rot = [1,2,1;0,0,0;-1,-2,-1];
imgBW = imgGreyResize > 150;

imgSobel = edge(imgGreyResize);
imgGreyResize_sobel = imfilter(imgGreyResize, sobel_filter_rot);

figure(2)
subplot(2,2,1);
imshow(imgGreyResizeNoiseG_f);
title('Gaussian Denoised');
subplot(2,2,2);
imshow(imgGreyResizeNoiseSP_f);
title('Salt and Pepper Denoised');
subplot(2,2,3);
imshow(imgGreyResize_sobel);
title('My sobel filter')
subplot(2,2,4);
imshow(imgSobel);
title('In built sobel filter');

