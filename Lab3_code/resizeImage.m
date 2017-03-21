function output = resizeImage(input)
    
    [h,w] = size(input);
    output = zeros(w);
    startPoint = int8((h-w)/2);
    for i = 1:w
        for j = 1:w
            output(j,i) = input(j,startPoint+i);
        end
    end
        
    
end
