%read cropped images and save them into 
fileName_training = 'images_cropped/training_set_cropped/*.png';
fileName_test = 'images_cropped/test_set_cropped/*.png';
path_training = 'images_cropped/training_set_cropped/';
path_test = 'images_cropped/test_set_cropped/';
path2save = 'cropped_images/';
files_training = dir(fullfile(pwd,fileName_training));
files_test = dir(fullfile(pwd,fileName_test));
training_images = [];
test_images = [];
N=size(files_training,1);
k = 10;

for i = 1:size(files_training,1)
    file_name = strcat(path_training, files_training(i,1).name);
    img = imread(file_name);
    img = rgb2gray(img);
    img = imresize(img,[200,200]);
    [irow, icol]=size(img);
    training_images = [training_images, img(:)];
end

for i = 1:size(files_test,1)
    file_name = strcat(path_test, files_test(i,1).name);
    img = imread(file_name);
    img = double(rgb2gray(img));
    img = imresize(img,[200,200]);
    [irow, icol]=size(img);
    test_images = [test_images, img(:)];
end
%obtain mean face.
meanFace=mean(training_images,2);
figure(3);
imshow(reshape(uint8(meanFace),icol,irow));
%obtain mean image.
meanImage=zeros(icol*irow, N);
for i=1:N
    meanImage(:,i) = double(training_images(:,i)) - meanFace;
end

%Covariance matrix
covariance = meanImage' * meanImage;
%covariance = covariance ./ N;
[eigenVector,eigenValue] = eigs(covariance,k);
eigenFaces = meanImage * eigenVector;

%show eigenfaces
figure(4);
for i=1:k
    img=reshape(eigenFaces(:,i),icol,irow);
    img = uint8(img);
    subplot(5,3,i)
    imshow(img)
    drawnow;
end


%Calculate K-coefficients of each training set.
eigenFaces = eigenFaces / norm(eigenFaces);
Kcoefficients = eigenFaces' * meanImage;

for n = 1:size(files_test,1);
    
    Kcoefficient = eigenFaces' * ((double(test_images(:,n)) - meanFace)); 
    reconstructImage = meanFace + eigenFaces(:,1:k)*Kcoefficient;
    %find 3 nearest neighbor
    id = nearest_neighbor(Kcoefficients, Kcoefficient);
    
    figure;
    subplot(1,5,1);
    imshow(reshape(uint8(test_images(:,n)),icol,irow));
    title('original')
    for i=1:3
        subplot(1,5,i+1)
        img = reshape(training_images(:,id(i)),icol,irow);
        imshow(uint8(img));
        title(i);
    end
    subplot(1,5,5);
    imshow(reshape(uint8(reconstructImage),icol,irow));
    title('reconstruction')
    
    disS = [];
    disp(n);
    for y=1:size(files_training,1)
        dis = sum((Kcoefficient - Kcoefficients(y)).^2);
        disS = [disS, dis];
    end
    disp(max(disS));
    disp(min(disS));  
    
end

my_face = imread('face_01_U5934839.jpg');
my_face = rgb2gray(my_face);
my_face = imresize(my_face,[200,200]);
Kcoefficient = eigenFaces' * ((double(my_face(:)) - meanFace));
reconstructImage = meanFace + eigenFaces(:,1:k)*Kcoefficient;

disp('my face');
disS = [];
disp(n);
for y=1:size(files_training,1)
    dis = sum((Kcoefficient - Kcoefficients(y)).^2);
    disS = [disS, dis];
end
disp(max(disS));
disp(min(disS));

my_face = imread('k2.jpg');
my_face = rgb2gray(my_face);
my_face = imresize(my_face,[200,200]);
Kcoefficient = eigenFaces' * ((double(my_face(:)) - meanFace));
reconstructImage = meanFace + eigenFaces(:,1:k)*Kcoefficient;

disp('flower');
disS = [];
disp(n);
for y=1:size(files_training,1)
    dis = sum((Kcoefficient - Kcoefficients(y)).^2);
    disS = [disS, dis];
end
disp(max(disS));
disp(min(disS));

%find 3 nearest neighbor
id = nearest_neighbor(Kcoefficients, Kcoefficient);

figure(100);
subplot(1,5,1);
imshow(reshape(my_face,icol,irow));
title('original')
for i=1:3
    subplot(1,5,i+1);
    img = reshape(training_images(:,id(i)),icol,irow);
    imshow(uint8(img));
    title(i);
end
subplot(1,5,5);
imshow(reshape(uint8(reconstructImage),icol,irow));
title('reconstructed')

%calculate and plot how much energy captured by k coefficient
total_energy = sum(Kcoefficient.^2);

figure,plot(cumsum((Kcoefficient .^ 2 / total_energy)*100));

title('Energy captured by first k principal component');

ylabel('percentage of captured energy');
xlabel('k principal component');