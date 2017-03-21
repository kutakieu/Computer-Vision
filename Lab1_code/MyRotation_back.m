function output = MyRotation_back(angleDeg,input_image)
%prepare output image
output = input_image .* 0;
angleDeg = angleDeg*(-1);
cos = cosd(angleDeg);
sin = sind(angleDeg);
[w,h] = size(input_image);
cx = w/2;
cy = h/2;
%calculate which coordinate of input image corresponds to each coordinates
%of output image
for i = 1 : w;
    for j = 1 : h;
        %choose nearest coordinates by casting with int16
        x = int16((i - cx)*cos - (j - cy)*sin + cx );
        y = int16((i - cx)*sin + (j - cy)*cos + cy );
        if x < w && x > 0 && y < h && y > 0 
          output(i,j) = input_image(x,y);
        end
    end
end
end