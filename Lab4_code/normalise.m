function uv_norm = normalise(uv, img)
    
    [h,w,c] = size(img);
    T = [w+h,0,w/2; 0,w+h,h/2; 0,0,1];
    T = inv(T);
    
    uv_norm = T * uv';
    uv_norm = uv_norm';

end