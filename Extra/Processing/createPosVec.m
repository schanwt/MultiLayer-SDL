function pos = createPosVec(poolSize, nrow)
pos = [];

if length(poolSize) == 1
    poolSize_1 = poolSize;
    poolSize_2 = poolSize;
else
    poolSize_1 = poolSize(1);
    poolSize_2 = poolSize(2);
end

for j = 0 : poolSize_2-1
    for i = 0 : poolSize_1-1
        pos = [pos i+j*nrow];
    end
end