theta = 30; len = 20;
fil = imrotate(ones(1, len), theta, 'bilinear');
fil = fil / sum(fil(:));
figure(2), im2 = imfilter(imgGreyResize, fil);
imshow(im2); 

PSF = fspecial('motion', len, theta);
%apply inbuilt wiener filter
wnr1 = deconvwnr(im2, PSF, 0.01);
%apply my wiener filter
im = myWiener_filter(im2, fil, 0.01);

subplot(221);
imshow(im2);
title('blured image');
subplot(222);
imshow(wnr1);
title('deblured image inbuilt Wiener filter');
subplot(223);
imshow(im);
title('deblured image myWiener filter');