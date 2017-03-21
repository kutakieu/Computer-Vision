%read cropped images and save them into 
fileName = 'images_cropped/training_set_cropped/*.png';
path = 'images_cropped/training_set_cropped/';
path2save = 'cropped_images/';
files = dir(fullfile(pwd,fileName));
training_images = {};

for i = 1:size(files,1)
    %someHandle = fopen(files(i,1).name);
    file_name = strcat(path, files(i,1).name);
    %filename2save = strrep(file_name, '.png', '');
    %filename2save = strcat(path2save, filename2save);
    %disp('images_cropped/training_set_cropped/'+name);
    temp = imread(file_name);
    temp = double(rgb2gray(temp));
    %h,w = size(temp);
    temp = resizeImage(temp);
    %imwrite(temp, filename2save);
    training_images = [training_images, temp];
end

d = eigs(temp,10);