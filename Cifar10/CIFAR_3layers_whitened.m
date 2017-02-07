%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example : Training of a 3-layer architecture on Cifar-10
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath ../Extra/speedup
addpath ../Extra/spams-matlab/
addpath ../Extra/spams-matlab/build/
addpath ../Extra/K_SVD/
addpath ../Extra/K_SVD/util/
addpath ../Extra/Processing/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here : Path to the dataset
addpath ./cifar-10-batches-mat/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the images of the training set
train_images = zeros(50000, 3072);
train_labels = zeros(10000,1);
load('data_batch_1.mat');
train_images(1:10000, :) = data;
train_labels (1:10000) = labels;
load('data_batch_2.mat');
train_images(10001:20000, :) = data;
train_labels (10001:20000) = labels;
load('data_batch_3.mat');
train_images(20001:30000, :) = data;
train_labels (20001:30000) = labels;
load('data_batch_4.mat');
train_images(30001:40000, :) = data;
train_labels (30001:40000) = labels;
load('data_batch_5.mat');
train_images(40001:50000, :) = data;
train_labels (40001:50000) = labels;

nsamples = size(train_images,1);

% Reshaping the images to 32 x 32 x 3 (color), dimensions 1 and 2 are
% swapped
train_images = reshape(train_images', [32 32 3, nsamples]);

% Swap the dimensions to correct
for i = 1 : size(train_images,4)
    train_images(:,:,:, i) = permute(train_images(:,:,:, i), [2 1 3]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Dictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load a set of dictionaries : they can be learned with
% "Learn_3layers_whitened_v2" or we can continue the learning of an old set 
% of dictionaries
load('CIFAR_3layers_18-07_50-50-100_whitened_ppatch_v2.mat');

%% Parameters
sname = 'CIFAR_3layers_23-11_25-25-50_whitened_ppatch_v2.mat';

% Regularization ||W||²
regulW = 0.0001;

if ~exist('batchsize', 'var')
    batchsize = 10;
end

if ~exist('lng_rate', 'var')
    lng_rate = 0.3;
end

if ~exist('lng_rate_0', 'var')
    lng_rate_0 = 0.3;
else
    lng_rate_0 = lng_rate;
end

if ~exist('param', 'var')
    param.lambda=0.1; % 0.05
    param.lambda2=0.01;
    param.mode = 2;
end

if ~exist('param2', 'var')
    param2.lambda=0.1; % 0.05
    param2.lambda2=0.01;
    param2.mode = 2;
end

if ~exist('param3', 'var')
    param3.lambda=0.1; % 0.05
    param3.lambda2=0.01;
    param3.mode = 2;
end

D_upg_gb_2 = zeros([size(D_2) batchsize]);
D_upg_gb_1 = zeros([size(D_1) batchsize]);
D_upg_gb_0 = zeros([size(D_0) batchsize]);

pass_max = floor(nsamples/batchsize);
it_counter = 1;
if ~exist('l_err', 'var')
    l_err= [];
end

for gb_iter = 1 : 30
    for inner_iter = 1 : pass_max
    disp(['iter - ' num2str(gb_iter)]);
    err = 0;
    err2 = 0;
    classif_res = 0;
    classif_res2 = 0;
    
    idx = [(inner_iter-1)*batchsize+1 : inner_iter*batchsize];  
    patch_list_tmp = train_images(:,:,:,idx);
    
    % Example of data augmentation (flipping)
    patch_list = zeros(size( patch_list_tmp,1), size( patch_list_tmp,2), size( patch_list_tmp,3), batchsize*2);
    currLabel = zeros(batchsize*2,1);
    nmctr = 0;
    for nm = 1:length(idx)
        tmp = patch_list_tmp(:, :, :, nm);
        patch_list(:,:,:, nmctr*2+1) =         tmp;
        patch_list(:,:,:, nmctr*2+2) = flipdim(tmp,2);
        currLabel(nmctr*2 +1 : (nmctr+1)*2) = repmat(train_labels(idx(nm)), [2 1]);
        nmctr = nmctr +1;
    end

    % Main loop
    parfor bs = 1 : size(patch_list,4)
        rec_Level2_list = [];
        rec_Level1_list = [];
        
        % Forward pass
        Level_1 = computeMultiScaleNextLayerPosWhiten(patch_list(:,:,:, bs), [5 5 3], [1 1 1], D_0, M, P, param);       
        pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);        
        Level_2 = computeMultiScaleNextLayerPos(pooling_1.Pooled, [5 5 size(D_0,2)], [1 1 1], D_1, param2); 
        pooling_2 = computePoolingLayerAvg_Fast_v2(max(Level_2.Layer, 0), 2);  
        Level_3 = computeMultiScaleNextLayerPos(pooling_2.Pooled, [5 5 size(D_1,2)], [1 1 1], D_2, param3);
        
        feats = [reshape(max(Level_3.Layer,0), [size(D_2,2) 1])];
        
        cL = zeros(10, 1);
        cL(currLabel(bs)+1, 1) = 1;
        
        % Softmax
        res_1 = [feats];
        p = exp( W'*res_1 )./repmat(sum(exp( W'*res_1 ), 1), [size(W,2) 1]);
        err = err + -sum(sum(cL.*log(p),1));
        [m, pr] = max(p);

        % Estimation of the accuracy on the training set
        if (cL(pr, 1)) == 1
            classif_res = classif_res +1;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Backpropagation
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % BP of the cost function
        dldX = W*(p - cL).*reshape(Level_3.Sign, [size(D_2,2) 1]);

        counter = 0;
        tmp_upg_gb_0 = zeros(size(D_0)); 
        
        % BP of layer 3
        [tmp_upg_gb_2, up_rec_Level2] = BP_Layer_Unpool(pooling_2.Pooled, ...
                                                        dldX, ...
                                                        [5 5 size(D_1,2)], ...
                                                        Level_3.xCode, ...
                                                        D_2, ...
                                                        Level_2.Sign, ...
                                                        param3);
        
        currCorr = im2colstep(up_rec_Level2, [1 1 size(up_rec_Level2,3)], [1 1 1]);
        
        % BP of layer 2
        [tmp_upg_gb_1, up_rec_Level1] = BP_Layer_Unpool(pooling_1.Pooled, ...
                                                        currCorr, ...
                                                        [5 5 size(D_0,2)], ...
                                                        Level_2.xCode, ...
                                                        D_1, ...
                                                        Level_1.Sign, ...
                                                        param2);
               
        currCorr = im2colstep(up_rec_Level1, [1 1 size(up_rec_Level1,3)], [1 1 1]);
               
        % BP of layer 1
        currPatch = im2colstep(patch_list(:,:,:,bs), [5 5 3], [1 1 1]);
        currPatchNorm = bsxfun(@rdivide, bsxfun(@minus, currPatch', mean(currPatch,1)'), sqrt(var(currPatch,[],1)'+10));
        currPatchNorm = (bsxfun(@minus, currPatchNorm, M) * P)';        

        for m = 1 : size(currPatch,2)
            currCode_L0 = Level_1.xCode(:, m);
            D_tilde = D_0(:, currCode_L0~=0);
            dDtD_inv = (D_tilde'*D_tilde + param.lambda2*eye(size(D_tilde,2)))\eye(size(D_tilde,2));            
            Beta_0 = zeros(size(D_0,2), 1);
            Beta_0(currCode_L0~=0, 1) = dDtD_inv*currCorr(currCode_L0~=0,m);         
            tmp_upg_gb_0 = tmp_upg_gb_0 + (-D_0*Beta_0*(currCode_L0') + (currPatchNorm(:,m) - D_0*currCode_L0)*(Beta_0'));
        end
         
        D_upg_gb_2(:,:, bs) = tmp_upg_gb_2;
        D_upg_gb_1(:,:, bs) = tmp_upg_gb_1;
        D_upg_gb_0(:,:, bs) = tmp_upg_gb_0;
        updW(:,:, bs) = res_1*(p - cL)';
    end
    disp('Errors : ');
    disp(classif_res/(size(patch_list,4)));
    disp(err/(size(patch_list,4)));
    
    % Average the gradients over the patches of the batch
    D_upg_gb_2_fnl = mean(D_upg_gb_2,3);
    D_upg_gb_1_fnl = mean(D_upg_gb_1,3);
    D_upg_gb_0_fnl = mean(D_upg_gb_0,3);
    updW_fnl = sum(updW,3)/size(patch_list,4);
 
    % Update
    D_2 = D_2 - lng_rate*D_upg_gb_2_fnl ;
    D_1 = D_1 - lng_rate*D_upg_gb_1_fnl ;
    D_0 = D_0 - lng_rate*D_upg_gb_0_fnl ;
    W = W - lng_rate*(updW_fnl + 2*regulW*W);
    
    % Normalization of the dictionaries
    D_0 = D_0./repmat(sqrt(sum(D_0.^2,1)), [size(D_0,1) 1]);
    D_0(isnan(D_0)) = 0;
    D_1 = D_1./repmat(sqrt(sum(D_1.^2,1)), [size(D_1,1) 1]);
    D_1(isnan(D_1)) = 0;
    D_2 = D_2./repmat(sqrt(sum(D_2.^2,1)), [size(D_2,1) 1]);
    D_2(isnan(D_2)) = 0;
    
    % Step-like weights for the gradients
    lng_rate = lng_rate_0/(2^floor(gb_iter/3)); 
  
    % Checking
    if (mod(inner_iter, 15) == 1)
        parfor bs = 1 : size(patch_list,4)
            
            % Forward pass
            Level_1 = computeMultiScaleNextLayerPosWhiten(patch_list(:,:,:, bs), [5 5 3], [1 1 1], D_0, M, P, param);
            pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);
            Level_2 = computeMultiScaleNextLayerPos(pooling_1.Pooled, [5 5 size(D_0,2)], [1 1 1], D_1, param2);
            pooling_2 = computePoolingLayerAvg_Fast_v2(max(Level_2.Layer, 0), 2);
            Level_3 = computeMultiScaleNextLayerPos(pooling_2.Pooled, [5 5 size(D_1,2)], [1 1 1], D_2, param3);
            feats = [reshape(max(Level_3.Layer,0), [size(D_2,2) 1])];
            
            cL = zeros(10, 1);
            cL(currLabel(bs)+1, 1) = 1;
            
            % Softmax
            res_1 = [feats];
            p = exp( W'*res_1 )./repmat(sum(exp( W'*res_1 ), 1), [size(W,2) 1]);
            err2 = err2 + -sum(sum(cL.*log(p),1));
            [m, pr] = max(p);
            
            % Estimation of the accuracy after the correction
            if (cL(pr, 1)) == 1
                classif_res2 = classif_res2 +1;
            end
            
        end
        disp('Corrected : ');
        disp(classif_res2/(size(patch_list,4)));
        disp(err2/(size(patch_list,4)));
    end
   
    if (mod(inner_iter, 5) == 0)
        save(sname, 'D_0', 'D_1', 'D_2', 'W', 'M', 'P', 'l_err', 'param', 'param2', 'param3', 'batchsize', 'nsamples', 'lng_rate', 'lng_rate_0');
    end

    it_counter = it_counter +1;
    l_err = [l_err classif_res/(size(patch_list,4))];
    end
end
display('End train');
%%
