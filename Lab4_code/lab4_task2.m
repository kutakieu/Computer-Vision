img_L = imread('Left.jpg');
img_R = imread('Right.jpg');

uv_L = load('uv_L.txt');
uv_R = load('uv_R.txt');

W = [1;1;1;1;1;1;1;1];
uv_L(:,3) = W;
uv_R(:,3) = W;

uv_R_norm = normalise(uv_R, img_R);

H = DLT(uv_L(:,1), uv_L(:,2), uv_R_norm(:,1), uv_R_norm(:,2));

[h,w,c] = size(img_R);
T = [w+h,0,w/2; 0,w+h,h/2; 0,0,1];

H = T*H;

x = H * uv_L';
x = x';

for i=1:size(x,1)
    re_uv(i,:) = x(i,:) ./ x(i,3);
end

figure(1)
imshow(img_R);
hold on;
plot(re_uv(:,1), re_uv(:,2), 'or');
title('reconstructed points');

figure(2)
imshow(img_L);
hold on;
plot(uv_L(:,1), uv_L(:,2), 'oc');