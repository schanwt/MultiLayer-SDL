function out = computePoolingLayerAvg_Fast_v2(patchlist, poolSize)
% out = computePoolingLayerAvg_Fast_v2(patchlist, poolSize)
% Input :
% - patchlist: input image
% - poolSize: size of the pooling step


out = struct('poolSize', [],  'Pooling', [], 'size', [], 'Pooled', []);

psize = [size(patchlist,1) size(patchlist,2)];
out(1).poolSize = poolSize;
out(1).Pooling = createPoolingGroup3D(size(patchlist), poolSize, poolSize);
out(1).size =  predRowCol(psize, poolSize, poolSize);
% Reshape in such a way that first 2 dimensions = coordinates of the
% original patches, 3rd dimension = coefficients of the decomposition

res_0        = zeros(size(out.Pooling,2), 1);
res_0 = mean( patchlist(out.Pooling), 1);

out(1).Pooled  =  reshape(res_0, [out(1).size size(patchlist,3)]);
