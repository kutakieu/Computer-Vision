I = imread('text.png');
I = imresize(I,[1024,1024]);
I = rgb2gray(I);
BW = I > 150;
countW = 0; countB = 0;
for i=1:1024;
    for j=1:1024;
        if BW(i,j)==0
            countB = countB + 1;
        else
            countW = countW + 1;
        end
    end
end



se = strel('line',3,0);
dilatedBW1 = imdilate(BW,se);
se = strel('line',5,0);
dilatedBW2 = imdilate(BW,se);
se = strel('line',9,0);
dilatedBW3 = imdilate(BW,se);

subplot(2,2,1)
imshow(dilatedBW1)
title('dilatedBW1')
subplot(2,2,2)
imshow(dilatedBW2)
title('dilatedBW2')
subplot(2,2,3)
imshow(dilatedBW3)
title('dilatedBW3')

se = strel('line',1024,0);
erodedBW1 = imerode(BW,se);
se = strel('line',1536,0);
erodedBW2 = imerode(BW,se);
se = strel('line',2048,0);
erodedBW3 = imerode(BW,se);
subplot(2,2,1)
imshow(erodedBW1)
title('erodedBW1')
subplot(2,2,2)
imshow(erodedBW2)
title('erodedBW2')
subplot(2,2,3)
imshow(erodedBW3)
title('erodedBW3')

se = strel('disk', 3);
closedBW1 = imclose(BW, se);
se = strel('disk', 5);
closedBW2 = imclose(BW, se);
se = strel('disk', 10);
closedBW3 = imclose(BW, se);
subplot(2,2,1)
imshow(closedBW1)
title('closedBW1')
subplot(2,2,2)
imshow(closedBW2)
title('closedBW2')
subplot(2,2,3)
imshow(closedBW3)
title('closedBW3')

se = strel('disk', 3);
openedBW1 = imopen(BW, se);
se = strel('disk', 5);
openedBW2 = imopen(BW, se);
se = strel('disk', 10);
openedBW3 = imopen(BW, se);
subplot(2,2,1)
imshow(openedBW1)
title('openedBW1')
subplot(2,2,2)
imshow(openedBW2)
title('openedBW2')
subplot(2,2,3)
imshow(openedBW3)
title('openedBW3')

line_images = {};
line_images = extract_lines(I);
for i = 1:size(line_images,2);
    subplot(2,3,i)
    imshow(line_images{i});
end

%input = I;
% BW = input > 150;
% se = strel('line',2*w,0);
% erodedBW = imerode(BW,se);
% imshow(erodedBW);