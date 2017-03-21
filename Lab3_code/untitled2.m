%eclose all;
%clear;
imgDim = [200, 200];
k = 20;
trainingDir = 'training_set/';
trainingFiles = dir(strcat(trainingDir,'*.png'));
nTrainingFiles = length(trainingFiles);
testingDir = 'test_set/';
testingFiles = dir(strcat(testingDir,'*.png'));
nTestingFiles = length(testingFiles);

images = zeros(prod(imgDim), nTrainingFiles);

% step 2
for i = 1:nTrainingFiles
    img = imread(strcat(trainingDir, trainingFiles(i).name));
    img = imresize(img, imgDim);
    img = rgb2gray(img);
    images(:,i) = img(:);
    
end

% step 3
% Calucate Mean 
meanFace = mean(images, 2);
meanImages = zeros(prod(imgDim), nTrainingFiles);
figure(10);
face = reshape(meanFace,200,200);
imshow(uint8(face));

for i = 1:nTrainingFiles
    meanImages(:,i) = images(:,i) - meanFace;
end

% B^t * B
covarianceMatrix = meanImages' * meanImages;
covarianceMatrix = covarianceMatrix ./ (nTrainingFiles - 1);
[eigVectors, eigValues] = eigs(covarianceMatrix, k);
eigVectors = meanImages * eigVectors;

% step 4
f = figure;
f.Name = 'Eigenfaces';
for n = 1:k
    subplot(2, ceil(k/2), n);
    %kk=eigVectors(:,n);
    %temp=sqrt(sum(kk.^2));
    %eigVectors(:,n)=eigVectors(:,n)./temp;
    eigFace = reshape(eigVectors(:,n), imgDim);
    eigFace = histeq(eigFace, 255);
    imshow(eigFace);
end
features = eigVectors' * meanImages;

% step 5
for i = 1:nTestingFiles
    img = imread(strcat(testingDir, testingFiles(i).name));
    img = imresize(img, imgDim);
    img = rgb2gray(img);
    feature = eigVectors' * (double(img(:)) - meanFace);
    dis = zeros(nTrainingFiles, 2);
    ReshapedImage = meanFace + eigVectors*feature; %m is the mean image, u is the eigenvector
    ReshapedImage = reshape(ReshapedImage,icol,irow);
    for j = 1:nTrainingFiles
       diff = features(:,j) - feature;
       dis(j,:) = [j sqrt(sum(diff.^2))];
    end
    sorted = sortrows(dis, 2);
    figure;
    subplot(1,4,1);
    imshow(ReshapedImage);
    title('test image')
    for p=1:3
        subplot(1,4,p+1);
        simg = images(:,sorted(p,1));
        simg = uint8(reshape(simg, imgDim));
        imshow(simg)
    end
end
% 
% 
% % step 6
% myface = imread(strcat(testingDir,'My_face_cropped.png'));
% myface = imresize(myface, imgDim);
% myface = rgb2gray(myface);
% feature = eigVectors' * (double(myface(:)) - meanFace);
% dis = zeros(nTrainingFiles, 2);
% for j = 1:nTrainingFiles
%     diff = features(:,j) - feature;
%     dis(j,:) = [j sqrt(sum(diff.^2))];
% end
% sorted = sortrows(dis, 2);
% figure;
% subplot(1,4,1);
% imshow(myface);
% title('test image')
% for p=1:3
%     subplot(1,4,p+1);
%     simg = images(:,sorted(p,1));
%     simg = uint8(reshape(simg, imgDim));
%     imshow(simg)
% end
% 
% 
% 
% % question 1 dont know how to caluclate the energy
% 
% % eigValues = diag(eigValues);
% % normEigValues = eigValues/sum(eigValues);
% % energy = eigValues .* feature;
% % figure, plot(cumsum(normEigValues));
% % % question 2
% 
% figure;
% reconstructFace = meanFace;
% for i = 1:k
%     temp = feature(i) * eigVectors(:,i);
%     reconstructFace = reconstructFace + temp;
%     subplot(2, ceil(k/2), i);
%     imshow(uint8(reshape(reconstructFace, imgDim)));
% end
% 
% 
