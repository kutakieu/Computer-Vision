function output = MyRotation_for(angleDeg,input_image)
[w,h] = size(input_image);
output = input_image .* 0;
cos = cosd(angleDeg);
sin = sind(angleDeg);
cx = w/2;
cy = h/2;
for i = 1 : w;
    for j = 1 : h;
        x = int16((i - cx)*cos - (j - cy)*sin + cx );
        y = int16((i - cx)*sin + (j - cy)*cos + cy );
        if x > 0 && y > 0 
          output(x,y) = input_image(i,j);
        end
    end
end
% 
% diagonal = w^2 + h^2;
% diagonal = sqrt(diagonal);
% newHeight = diagonal * sin;
% newWidth = diagonal * cos;
% pad_v = (newHeight - h)/2;
% pad_h = (newWidth - w)/2;
% PadImage = padarray(input_image, [pad_v, pad_h]);
% 
% cx = newHeight/2;
% cy = newWidth/2;
% for i = 1 : newHeight;
%     for j = 1 : newWidth;
%         x = int16((i - cx)*cos - (j - cy)*sin + cx );
%         y = int16((i - cx)*sin + (j - cy)*cos + cy );
%         if x > 0 && y > 0 
%           output(x,y) = PadImage(i,j);
%         end
%     end
% end
% 

end