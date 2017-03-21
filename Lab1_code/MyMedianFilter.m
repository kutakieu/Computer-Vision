function output_image = MyMedianFilter(input_image)
%prepare output image
output_image = 0.*input_image;
[isizev,isizeh] = size(input_image);
%pad around input image
PadImage = padarray(input_image, [3,3]);

for i = 2 : (isizev - 1)
    for j = 2 : (isizeh - 1)
        %extract 3 by 3 matrix from input image
        filter = PadImage(((i-1):(i+1)) , ((j-1):(j+1)));
        %pick out median value of this matrix
        output_image(i,j) = median(filter(:));
    end
end
output_image = uint8(output_image);
end