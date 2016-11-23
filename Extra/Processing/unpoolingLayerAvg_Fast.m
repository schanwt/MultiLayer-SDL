function out = unpoolingLayerAvg_Fast(pooledLayer, poolSize)
% out = unpoolingLayerAvg_Fast(pooledLayer, poolSize, psize)
% Input :
% - pooledLayer: input patches, as columns
% - poolSize: size of the pooling step

out = zeros(poolSize*size(pooledLayer,1), poolSize*size(pooledLayer,2), size(pooledLayer, 3));

pGroup = createPoolingGroup3D(size(out), poolSize, poolSize);

for j = 1 : size(pGroup,2)
    idx = pGroup(:,j);
    out(idx) = pooledLayer(j)/(poolSize^2);
end

