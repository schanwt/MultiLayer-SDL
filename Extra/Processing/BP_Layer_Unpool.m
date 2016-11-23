function [tmp_upg_gb, up_rec_Level] = BP_Layer_Unpool(input, currCorr, psize, xcodes, current_D, outSign, param)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backpropagation function through a layer
% input : input "image"
% currCorr : matrix which contains the error values assiociated with each
%           patch
% psize : patch size
% xcodes : Sparse codes of the patches (contained in the structure Level_X,
%           ex : Level_1.xCode)
% current_D : Dictionary
% outSign : Sign of the coefficients (for the ReLU, contained in
%           the structure Level_X, ex : Level_1.Sign)
% param : parameters used to compute the sparse codes (see SPAMS toolbox)
%%% Sorties :
% - tmp_upg_gb : gradients of the dictionary with respect to the "input"
% - up_rec_Level : Error "image" (this error is used to continue the
%                   backpropagation through the previous layer)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currPatch = im2colstep(input, psize, [1 1 1]);
currPatchNorm = currPatch./repmat(sqrt(sum(currPatch.^2,1)), [size(currPatch,1) 1]);
currPatchNorm(isnan(currPatchNorm)) = 0;

tmp_upg_gb = zeros(size(current_D));

for m = 1 : size(currPatch,2)
    currCode = xcodes(:,m);
    D_tilde = current_D(:, currCode~=0);    
    dDtD_inv = (D_tilde'*D_tilde + param.lambda2*eye(size(D_tilde,2)))\eye(size(D_tilde,2));

    Beta = zeros(size(current_D,2), 1);
    Beta(currCode~=0, 1) = dDtD_inv*currCorr(currCode~=0,m);

    tmp_upg_gb = tmp_upg_gb + (-current_D*Beta*(currCode') + (currPatchNorm(:,m) - current_D*currCode)*(Beta'));

    dX2curr_dXprev = zeros(size(current_D,2), size(current_D,1));
    dX2curr_dXprev(currCode~=0, :) = dDtD_inv*D_tilde';

    mden = (sum(currPatch(:,m).^2)*sqrt(sum(currPatch(:,m).^2)));
    if (mden == 0)
        mden = 1;
    end
    sqsum = sum(currPatch(:,m).^2);
    corrNorm = (-currPatch(:,m)*currPatch(:,m)' + diag(repmat(sqsum, [size(currPatch,1) 1])))/mden;

    rec_Level_list(:,m) = (dX2curr_dXprev(currCode~=0, :)'*currCorr(currCode~=0,m))'*corrNorm;    
end

rec_Level = col2imstep(rec_Level_list, size(input), psize, [1 1 1]);
up_rec_Level = unpoolingLayerAvg_Fast(rec_Level, 2);
up_rec_Level = up_rec_Level.*outSign;
