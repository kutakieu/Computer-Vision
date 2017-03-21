img = imread('stereo2012a.jpg');
uv = load('uv_new.txt');
XYZ = load('XYZ_new.txt');

W = [1;1;1;1;1;1];
uv(:,3) = W;

uv_norm = normalise(uv, img);

W = [1;1;1;1;1;1];

XYZ(:,4) = W;

C = calibrate(img, XYZ, uv_norm);

[h,w,c] = size(img);
T = [w+h,0,w/2; 0,w+h,h/2; 0,0,1];

C = T*C;

x = C * XYZ';
x = x';
re_uv = []
for i=1:6
    re_uv(i,:) = x(i,:) ./ x(i,3);
end

[K,R,t] = vgg_KR_from_P(C, 1);

error = sqrt(sum(sum((re_uv - uv).*(re_uv - uv))));

figure(1)
imshow(img);
hold on;
plot(re_uv(:,1), re_uv(:,2), 'or');
plot(uv(:,1), uv(:,2), 'oc');
title('reconstructed points');

origin = [0,0,0,1];
origin_2D = C * origin' ;

X = [1000,0,0,1];
X_2D = C * X';
X_2D = X_2D ./ X_2D(end,end);

Y = [0,1000,0,1];
Y_2D = C * Y';
Y_2D = Y_2D ./ Y_2D(end,end);

Z = [0,0,1000,1];
Z_2D = C * Z';
Z_2D = Z_2D ./ Z_2D(end,end);

plot([origin_2D(1,1),X_2D(1,1)],[origin_2D(2,1),X_2D(2,1)],'Color','r','LineWidth',1);
plot([origin_2D(1,1),Y_2D(1,1)],[origin_2D(2,1),Y_2D(2,1)],'Color','r','LineWidth',1);
plot([origin_2D(1,1),Z_2D(1,1)],[origin_2D(2,1),Z_2D(2,1)],'Color','r','LineWidth',1);