I = imread('text.png');
I = imresize(img,[1024,1024]);
I = rgb2gray(I);
angleDeg = 45;
Irotate = imrotate(I, angleDeg, 'bilinear', 'crop');

imgMyRotation1 = MyRotation_for(45, I);

imgMyRotation2 = MyRotation_back(45, I);

subplot(131);
imshow(imgMyRotation1);
title('forward')
subplot(132);
imshow(imgMyRotation2);
title('backward')
subplot(133);
imshow(Irotate);
title('inbuilt rotate')