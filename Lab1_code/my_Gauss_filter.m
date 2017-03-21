function output_image =  my_Gauss_filter( noisy_image, my_9x9_gausskernel)

output_image = noisy_image.*0;

for i=5:507;
for j=5:507;
    output_image(i,j)=0;
    for k=-4:4;
        for l=-4:4;
            output_image(i,j) = output_image(i,j)+noisy_image(i+k,j+l)*my_9x9_gausskernel(k+5,l+5);
        end
    end
end
end
end