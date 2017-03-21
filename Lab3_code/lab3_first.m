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
um=100;
ustd=80;
M=size(files_training,1);

for i = 1:size(files_training,1)
    file_name = strcat(path_training, files_training(i,1).name);
    img = imread(file_name);
    img = double(rgb2gray(img));
    img = imresize(img,[200,200]);
    %img = resizeImage(img);
    [irow, icol]=size(img);
    temp=reshape(img',irow*icol,1);
    training_images = [training_images, temp];
end

for i = 1:size(files_test,1)
    file_name = strcat(path_test, files_test(i,1).name);
    img = imread(file_name);
    img = double(rgb2gray(img));
    img = imresize(img,[200,200]);
    %img = resizeImage(img);
    [irow, icol]=size(img);
    temp=reshape(img',irow*icol,1);
    test_images = [training_images, temp];
end

for i=1:size(training_images,2)
    temp=double(training_images(:,i));
    m=mean(temp);
    st=std(temp);
    training_images(:,i)=(temp-m)*ustd/st+um;
end

m=mean(training_images,2);  % obtains the mean of each row instead of each column
tmimg=uint8(m); % converts to unsigned 8-bit integer. Values range from 0 to 255
img=reshape(tmimg,icol,irow); % takes the N1*N2x1 vector and creates a N1xN2 matrix
img=img'; 
figure(3);
imshow(img);

% Change image for manipulation
dbx=[];    % A matrix
for i=1:M
temp=double(training_images(:,i));
dbx=[dbx temp];
end

%Covariance matrix C=A'A, L=AA'
A=dbx';
L=A*A';
% vv are the eigenvector for L
% dd are the eigenvalue for both L=dbx'*dbx and C=dbx*dbx';
[vv dd]=eig(L);
% Sort and eliminate those whose eigenvalue is zero
v=[];
d=[];
for i=1:size(vv,2)
if(dd(i,i)>1e-4)
v=[v vv(:,i)];
d=[d dd(i,i)];
end
end


%sort, will return an ascending sequence
[B index]=sort(d);
ind=zeros(size(index));
dtemp=zeros(size(index));
vtemp=zeros(size(v));
len=length(index);
for i=1:len
dtemp(i)=B(len+1-i);
ind(i)=len+1-index(i);
vtemp(:,ind(i))=v(:,i);
end
d=dtemp;
v=vtemp;


%Normalization of eigenvectors
for i=1:size(v,2) %access each column
kk=v(:,i);
temp=sqrt(sum(kk.^2));
v(:,i)=v(:,i)./temp;
end

%Eigenvectors of C matrix
u=[];
for i=1:size(v,2)
temp=sqrt(d(i));
u=[u (dbx*v(:,i))./temp];
end

%Normalization of eigenvectors
for i=1:size(u,2)
kk=u(:,i);
temp=sqrt(sum(kk.^2));
u(:,i)=u(:,i)./temp;
end

% show eigenfaces
figure(4);
for i=1:4
img=reshape(u(:,i),icol,irow);
img=img';
img=histeq(img,255);
subplot(2,2,i)
imshow(img)
drawnow;
if i==3
title('Eigenfaces','fontsize',18)
end
end

% Find the weight of each face in the training set
omega = [];
for h=1:size(dbx,2)
WW=[]; 
for i=1:size(u,2)
t = u(:,i)'; 
WeightOfImage = dot(t,dbx(:,h)');
WW = [WW; WeightOfImage];
end
omega = [omega WW];
end

%Calculate K-coefficients of each training set.
Kcoefficients = [];
%aa=size(u,2);
aa = 70;
for i=1:M
    temp=double(training_images(:,i));
    me=mean(temp);
    st=std(temp);
    %temp=(temp-me)*ustd/st+um;
    temp=(temp-me);
    NormImage = temp;
    %Difference = temp-m;
    p = [];
    for j = 1:aa
        pare = dot(NormImage,u(:,j));
        p = [p; pare];
    end
    Kcoefficients = [Kcoefficients, p];
end 

for x = 1:10
temp=double(test_images(:,x));
me=mean(temp);
st=std(temp);
%temp=(temp-me)*ustd/st+um;
temp=(temp-me);
NormImage = temp;
Difference = temp-m;

p = [];
for i = 1:aa
pare = dot(NormImage,u(:,i));
p = [p; pare];
end
ReshapedImage = m + u(:,1:aa)*p; %m is the mean image, u is the eigenvector
ReshapedImage = reshape(ReshapedImage,icol,irow);
ReshapedImage = uint8(ReshapedImage');
figure;
imshow(ReshapedImage);

%find 3 nearest neighbor
near1=inf;
near2=inf;
near3=inf;
id = [0,0,0];
for i = 1:M
    distance = sqrt((p .* Kcoefficients(i)).^2);
    distance = sum(distance);
    if near1 > distance
        near3 = near2;  id(3) = id(2);
        near2 = near1;  id(2) = id(1);
        near1 = distance;   id(1) = i;
        
    elseif near2 > distance
        near3 = near2;  id(3) = id(2);
        near2 = distance;   id(2) = i;
        
    elseif near3 > distance
        near3 = distance;   id(3) = i;
    end
end

figure(100);
subplot(2,2,1)
imshow(ReshapedImage)
title('original')
for i=1:3
    subplot(2,2,i+1)
    img = reshape(training_images(:,id(i)),icol,irow);
    imshow(uint8(img'));
    title(i);
end
end