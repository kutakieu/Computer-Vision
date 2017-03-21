function H = DLT(u2Trans, v2Trans, uBase, vBase)
% Computes the homography H applying the Direct Linear Transformation
% The transformation is such that
% p = H p' , i.e.,:
% (uBase, vBase, 1)'=H*(u2Trans , v2Trans, 1)'
%
% INPUTS:
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p')
% uBase, vBase - vectors with coordinates u and v of the original base image point p %
% OUTPUT
% H - a 3x3 Homography matrix %
% Taku Ueki , u5934839

    M = [];

    for i=1:size(u2Trans,1)
        temp = zeros(2,9);
        temp(1,4) = -u2Trans(i);
        temp(1,5) = -v2Trans(i);
        temp(1,6) = -1;
        
        temp(1,7) = vBase(i)*u2Trans(i);
        temp(1,8) = vBase(i)*v2Trans(i);
        temp(1,9) = vBase(i);
        
        temp(2,1) = u2Trans(i);
        temp(2,2) = v2Trans(i);
        temp(2,3) = 1;
        
        temp(2,7) = -uBase(i)*u2Trans(i);
        temp(2,8) = -uBase(i)*v2Trans(i);
        temp(2,9) = -uBase(i);
        M = [M;temp];
    end
    
    [U,S,V] = svd(M);
    p = V(:,end);
    p = p ./ norm(p);
    H = reshape(p, 3,3);
    H = H';
    H = H ./ H(3,3);
end