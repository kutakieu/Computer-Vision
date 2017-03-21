%function applying input filter(my_filter) to input image.
function output =  my_filter( input_image, my_filter)
%prepare the output image
output = 0.*input_image;

[filsizev,filsizeh] = size(my_filter);
filsizev_half = (filsizev - 1)/2;
filsizeh_half = (filsizeh - 1)/2;

%pad around the input image size of half of filter size
PadImage = padarray(input_image, [filsizeh_half, filsizeh_half]);

[isizev,isizeh] = size(input_image);

for i = 1 + filsizev_half : isizev - filsizev_half;
    for j = 1 + filsizeh_half : isizeh - filsizeh_half;
        %extract each matrix from input image and convolute with filter
        img_matrix = PadImage(i-filsizev_half : i-filsizev_half+filsizev-1,...
            j - filsizeh_half : j-filsizeh_half + filsizeh-1);
        temp = double(img_matrix).*my_filter;
        output (i,j)=sum(temp(:));        
    end
end
output = uint8(output);
end