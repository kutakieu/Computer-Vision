img2 = imread('colorwheel.jpg');
imgHSV = rgb2hsv(img2);
imshow(imgHSV)

H = imgHSV(:,:,1);
S = imgHSV(:,:,2);
V = imgHSV(:,:,3);

[COUNTS,X]=imhist(H, 50)
[COUNTS,Y]=imhist(H, 50)

subplot(2,2,1);
imshow(imgHSV)
subplot(2,2,2);
imshow(H)
subplot(2,2,3);
imshow(S)
subplot(2,2,4);
imshow(V)

imshow(H)
h = text(123,325,'0.2651')
h.Color = 'r';
h = text(268,145,'0.1902')
h.Color = 'r';
h = text(496,84,'0.1667')
h.Color = 'r';
h = text(712,147,'0.1257')
h.Color = 'r';
h = text(870,319,'0.1011')
h.Color = 'r';
h = text(925,550,'0.05144')
h.Color = 'r';
h = text(855,720,'0.01681')
h.Color = 'r';
h = text(714,900,'0.9486')
h.Color = 'r';
h = text(492,958,'0.7933')
h.Color = 'r';
h = text(279,891,'0.729')
h.Color = 'r';
h = text(141,739,'0.6206')
h.Color = 'r';
h = text(75,550,'0.549')
h.Color = 'r';

color = []
his = H(1,1)
count = 0
for row = 1:size(H,1)
    for column = 1:size(H,2)
        if H(row,col) == his
            count += 1
            if count == 10
                for i in 1:length(color) 
                    find(color,)
                        break;
                    end
                    text(row,col,his)
                    color = color + his
                end
                count = 0
            end
        else
            his = H(row,col)
        end
    end
end
            