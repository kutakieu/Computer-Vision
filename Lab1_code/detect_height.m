function height = detect_height(input)
top = 0;
bottom = 0;
counter = 0;
[h,w] = size(input);
for i = 0 : h;
    for j = 0 : w;
        if input(i, j) == 0;
            top = j;
        end
    end
end
for i = top : h;
    for j = 0 : w;
        if input(i, j) == 0;
            detect = true;
            break;
        end
    end
    if ~detect
        counter = counter + 1;
        if counter > 5
            bottom = i;
        end
    end
end
height = bottom - top;
end