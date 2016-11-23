function fList = createPoolingGroup3D(gridSize, poolSize, stride)
% fList = createPoolingGroup3D(gridSize, poolSize, stride)
% Create a list which contains the index of the pooled patches given
% poolSize and stride.
% Let's consider an image decomposed into 9 patches :
%  - - -
% |1 4 7| A 2x2 pooling with stride = 1 gives (columns): (1,2,4,5), 
% |2 5 8| (2,3,5,6), (4,5,7,8), (5,6,8,9).
% |3 6 9| A 2x2 pooling with stride = 2 gives (columns): (1,2,4,5).
%  - - -
% Input :
% - gridSize : number of patches [rows, columns]
% - poolSize: size of the pooling neighbourhood
% - stride: stride of the pooling 

% Predict the size after the pooling
nsize = predRowCol(gridSize, poolSize, stride);

n_row = nsize(1);
n_col = nsize(2);
n_depth = gridSize(3);

pos = createPosVec(poolSize, gridSize(1))';
pList = [];
for j = 1 : n_col
    for i = 1 : n_row
        pList = [pList (1+(i-1)*stride+(j-1)*stride*gridSize(1))+pos];
    end
end


%fList = [];
fList = zeros(size(pList,1), size(pList,2)*n_depth);

for k = 1 : n_depth
    fList(:,(k-1)*size(pList,2)+1 : k*size(pList,2)) = pList+(k-1)*gridSize(1)*gridSize(2);
end
