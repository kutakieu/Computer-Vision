function output = myWiener_filter(input, fil, K)

[h,w] = size(input);
%apply fft2 to input image
H = fft2(input);
%apply fft2 to input filter
Hfil = fft2(fil, size(input,1), size(input,2));
%calculate MMSE
for i = 1:h
    for j = 1:w
        Hfil(i,j) = conj(Hfil(i,j))/((norm(Hfil(i,j)))^2 + K);
    end
end
%apply wiener filter to input image
Hfill = H .* Hfil;
%apply ifft2 to recover the image
output = uint8(ifft2(Hfill));
