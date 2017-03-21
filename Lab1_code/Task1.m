img1 = imread('face _01_U5934839.jpg');
img1 = imresize(img1,[1024,720]);
imshow(img1)
resizeImg1 = imresize(img1,[768,512]);
img1Gray = rgb2gray(img1);
R1 = img1(:,:,1);
G1 = img1(:,:,2);
B1 = img1(:,:,3);

subplot(2,2,1)
imhist(R1)
subplot(2,2,2)
imhist(G1)
subplot(2,2,3)
imhist(B1)

imgEqR1 = histeq(R1)
imgEqG1 = histeq(G1)
imgEqB1 = histeq(B1)

subplot(2,2,1)
imhist(imgEqR1)
subplot(2,2,2)
imhist(imgEqG1)
subplot(2,2,3)
imhist(imgEqB1)




%img2 = imread('face _02_U5934839.jpg');
%img2 = imresize(img2,[1024,720]);
%imshow(img2)
resizeImg2 = imresize(img2,[768,512]);
img1Gray = rgb2gray(img2);
R2 = img2(:,:,1);
G2 = img2(:,:,2);
B2 = img2(:,:,3);

subplot(2,2,1)
imhist(R2)
subplot(2,2,2)
imhist(G2)
subplot(2,2,3)
imhist(B2)

imgEqR2 = histeq(R2)
imgEqG2 = histeq(G2)
imgEqB2 = histeq(B2)

subplot(2,2,1)
imhist(imgEqR2)
subplot(2,2,2)
imhist(imgEqG2)
subplot(2,2,3)
imhist(imgEqB2)

%img3 = imread('face _03_U5934839.jpg');
%img3 = imresize(img3,[1024,720]);
%imshow(img3)
resizeImg3 = imresize(img3,[768,512]);
img3Gray = rgb2gray(img3);
R3 = img3(:,:,1);
G3 = img3(:,:,2);
B3 = img3(:,:,3);

subplot(2,2,1)
imhist(R3)
subplot(2,2,2)
imhist(G3)
subplot(2,2,3)
imhist(B3)

imgEqR3 = histeq(R3)
imgEqG3 = histeq(G3)
imgEqB3 = histeq(B3)

subplot(2,2,1)
imhist(imgEqR3)
subplot(2,2,2)
imhist(imgEqG3)
subplot(2,2,3)
imhist(imgEqB3)


subplot(2,2,1)
imshow(resizeImg1)
subplot(2,2,2)
imshow(resizeImg2)
subplot(2,2,3)
imshow(resizeImg3)