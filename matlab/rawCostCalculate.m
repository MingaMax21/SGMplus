function rawCost = rawCostCalculate(imgL,imgR,dispLevels)
H = size(imgL,1);
W = size(imgR,2);
rawCost = zeros(dispLevels,H,W);
for d=1:dispLevels
    for i = 1:H
        for j =1:W
             if (j<=d)  
                 continue;
             end
             rawCost(d,i,j) = abs( double( imgL(i,j) ) - double(imgR(i,j-d) )  );
        end
    end

end