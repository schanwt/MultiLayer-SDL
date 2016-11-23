function out = computeMultiScaleNextLayerPos(currIm, patchSize, stride, D, param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the sparse codes of a single layer
% currIm : "image" to process
% patchSize : patch size
% stride : stride applied between each patch
% D : dictionary
% param : parameters used for the sparse coding (see doc spams, mexLasso)
%%% Output :
% out : structure which contains
% - psize : size of the output feature image
% - xCode : sparse codes of the layer (column-wise)
% - Layer : Feature "image" (after the concatenation of the codes)
% - Sign : signs, used in the BP of the ReLU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plist = im2colstep(currIm, patchSize, stride);
plist = plist./repmat(sqrt(sum(plist.^2,1)), [size(plist,1) 1]);
plist(isnan(plist)) = 0;

out.psize = predRowCol(size(currIm), patchSize(1), stride(1));
out.xCode = full(mexLasso(plist, D, param));
out.Layer = reshape(out.xCode', [out.psize(1) out.psize(2) size(D,2)]);
out.Sign = double(out.Layer > 0);