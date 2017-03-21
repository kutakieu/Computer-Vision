function line_images = extract_lines(input)
%prepare the arrays for store Y coordinates of each line's top and bottom
tops = {}; bottoms = {};
[h,w] = size(input);
%create Black and White image of input image.
BW = input > 150;
%obtain five line by eroding. kernel width is product of 2 and width of input image
se = strel('line',2*w,0);
erodedBW = imerode(BW,se);
imshow(erodedBW);

%store the Y coordinates which change from white to black as each line's top
%store the Y coordinates which change from black to white as bottom
for i = 2 : h;
    if erodedBW(i,1) == 0 && erodedBW(i-1, 1) == 1;
       tops{end+1} = i; 
    end
    if erodedBW(i,1) == 1 && erodedBW(i-1, 1) == 0;
       bottoms{end+1} = i; 
    end  
end

line_images = {};
%extract lines from stored each line's top and bottom
for i = 1 : size(tops,2);
    line_image = 0 .* input;
    for j = tops{i} : bottoms{i};
        for k = 1 : w;
            line_image(j, k) = input(j, k);
        end
    end
    line_images{end+1} = line_image;
end

end