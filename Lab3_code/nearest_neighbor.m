function id = nearest_neighbor(Kcoefficients, Kcoefficient)

near1=inf;
near2=inf;
near3=inf;
id = [0,0,0];
for i = 1:size(Kcoefficients,2)
    distance = sqrt((Kcoefficient - Kcoefficients(:,i)).^2);
    distance = sum(distance);
    if near1 > distance
        near3 = near2;  id(3) = id(2);
        near2 = near1;  id(2) = id(1);
        near1 = distance;   id(1) = i;
    elseif near2 > distance
        near3 = near2;  id(3) = id(2);
        near2 = distance;   id(2) = i;
    elseif near3 > distance
        near3 = distance;   id(3) = i;
    end
end
end