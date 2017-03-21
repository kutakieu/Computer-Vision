% CALIBRATE
%
% Function to perform camera calibration
%
% Usage:
%
%   Where:
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of
%  the calibration target.  It also projects the XYZ coordinates
%  back into image coordinates using the  calibration matrix
%  and plots these points too as a visual check on the accuracy of
%  the calibration process.
%  Lines from the origin to the vanishing points in the X, Y and
%  Z directions are overlaid on the image.
%  The mean squared error between the  positions of the uv
%coordinates and the projected XYZ coordinates is also reported.
%
function C = calibrate(im, XYZ, uv)
    M = [];

    for i=1:size(uv,1)
        temp = zeros(2,12);
        temp(1,5) = -XYZ(i,1);
        temp(1,6) = -XYZ(i,2);
        temp(1,7) = -XYZ(i,3);
        temp(1,8) = -XYZ(i,4);
        
        temp(1,9) = uv(i,2)*XYZ(i,1);
        temp(1,10) = uv(i,2)*XYZ(i,2);
        temp(1,11) = uv(i,2)*XYZ(i,3);
        temp(1,12) = uv(i,2);
        
        temp(2,1) = XYZ(i,1);
        temp(2,2) = XYZ(i,2);
        temp(2,3) = XYZ(i,3);
        temp(2,4) = XYZ(i,4);
        
        temp(2,9) = -uv(i,1)*XYZ(i,1);
        temp(2,10) = -uv(i,1)*XYZ(i,2);
        temp(2,11) = -uv(i,1)*XYZ(i,3);
        temp(2,12) = -uv(i,1);
        M = [M;temp];
    end
    
    [U,S,V] = svd(M);
    p = V(:,end);
    p = p ./ norm(p);
    C = reshape(p, 4,3);
    C = C';
    C = C ./ C(end,end);
end