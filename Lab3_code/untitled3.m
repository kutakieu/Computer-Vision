img = imread('my_face4_o.jpg');

%img = rgb2gray(img);

img = imrotate(img,-90);

img = imresize(img,0.25);

img_c = img(150:620, 170:540,:);

imwrite(img_c, 'my_face4.png')

imshow(img_c);

%img = imread('My_face_cropped.jpg');