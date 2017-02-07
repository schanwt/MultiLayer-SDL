%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example : Training of a 3-layer architecture on MNIST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath ../Extra/speedup
addpath ../Extra/spams-matlab/
addpath ../Extra/spams-matlab/build/
addpath ../Extra/K_SVD/
addpath ../Extra/K_SVD/util/
addpath ../Extra/Processing/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here : Path to the dataset
addpath ./Dataset/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train

% train_images = loadMNISTImages('train-images.idx3-ubyte');
% train_labels = loadMNISTLabels('train-labels.idx1-ubyte');

load('MNIST_1000_perClass_1.mat', 'train_images', 'train_labels');

patch_list_tmp = double(train_images);
patch_list_tmp = reshape(patch_list_tmp, [28 28 size(train_images,2)]);
patch_list_tmp = imresize(patch_list_tmp, [32 32]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Dictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load a set of dictionaries : they can be learned with
% "Learn_MNIST_3layers_whitened" or we can continue the learning of an old set 
% of dictionaries

%load('MNIST_3layers_22-11_Learning_25-25-50_1000perClass.mat');
load('MNIST_3layers_23-11_25-25-50_1000perClass.mat');

%% Parameters
sname = 'MNIST_3layers_23-11_25-25-50_1000perClass.mat';

% Regularization parameter ||W||²
regulW = 0.0005;
nsamples = size(train_images,2);

if ~exist('batchsize', 'var')
    batchsize = 10;
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
    
    patch_list = patch_list_tmp(:,:,idx);
    currLabel = train_labels(idx);
    
    parfor bs = 1 : size(patch_list,3)
        rec_Level2_list = [];
        rec_Level1_list = [];
        up_rec_Level2 = [];
        
        % Forward Pass
        Level_1 = computeMultiScaleNextLayerPos(patch_list(:,:, bs), [5 5], [1 1], D_0, param); 
        pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);
        Level_2 = computeMultiScaleNextLayerPos(pooling_1.Pooled, [5 5 size(D_0,2)], [1 1 1], D_1, param);
        pooling_2 = computePoolingLayerAvg_Fast_v2(max(Level_2.Layer, 0), 2);
        Level_3 = computeMultiScaleNextLayerPos(pooling_2.Pooled, [5 5 size(D_1,2)], [1 1 1], D_2, param);
        
        feats = [reshape(max(Level_3.Layer,0), [size(D_2,2) 1])];
        
        cL = zeros(10, 1);
        cL(currLabel(bs)+1, 1) = 1;
        
        %%% Softmax
        res_1 = [feats];
        p = exp( W'*res_1 )./repmat(sum(exp( W'*res_1 ), 1), [size(W,2) 1]);
        err = err + -sum(sum(cL.*log(p),1));
        [m, pr] = max(p);
        
        if (cL(pr, 1)) == 1
            classif_res = classif_res +1;
        end

        %%% Backpropagation
        dldX = W*(p - cL).*reshape(Level_3.Sign, [size(D_2,2) 1]);

        counter = 0;
        tmp_upg_gb_0 = zeros(size(D_0)); 
        
        % BP of Layer 3
        [tmp_upg_gb_2, up_rec_Level2] = BP_Layer_Unpool(pooling_2.Pooled, ...
                                                        dldX, ...
                                                        [5 5 size(D_1,2)], ...
                                                        Level_3.xCode, ...
                                                        D_2, ...
                                                        Level_2.Sign, ...
                                                        param3);
        
        currCorr = im2colstep(up_rec_Level2, [1 1 size(up_rec_Level2,3)], [1 1 1]);
        
        % BP of Layer 2
        [tmp_upg_gb_1, up_rec_Level1] = BP_Layer_Unpool(pooling_1.Pooled, ...
                                                        currCorr, ...
                                                        [5 5 size(D_0,2)], ...
                                                        Level_2.xCode, ...
                                                        D_1, ...
                                                        Level_1.Sign, ...
                                                        param2);
               
        currCorr = im2colstep(up_rec_Level1, [1 1 size(up_rec_Level1,3)], [1 1 1]);
        
        % BP of Layer 1       
        currPatch = im2colstep(patch_list(:,:,bs), [5 5], [1 1]);
        currPatchNorm = currPatch./repmat(sqrt(sum(currPatch.^2,1)), [size(currPatch,1) 1]);
        currPatchNorm(isnan(currPatchNorm)) = 0;
        
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
    disp(classif_res/(size(patch_list,3)));
    disp(err/(size(patch_list,3)));
    
    D_upg_gb_2_fnl = mean(D_upg_gb_2,3);
    D_upg_gb_1_fnl = mean(D_upg_gb_1,3);
    D_upg_gb_0_fnl = mean(D_upg_gb_0,3);
    updW_fnl = sum(updW,3)/size(patch_list,4);

    % Update
    D_2 = D_2 - lng_rate*D_upg_gb_2_fnl ;
    D_1 = D_1 - lng_rate*D_upg_gb_1_fnl ;
    D_0 = D_0 - lng_rate*D_upg_gb_0_fnl ;
    W = W - 2*lng_rate*(updW_fnl + 2*regulW*W);
    
    D_0 = D_0./repmat(sqrt(sum(D_0.^2,1)), [size(D_0,1) 1]);
    D_0(isnan(D_0)) = 0;
    
    D_1 = D_1./repmat(sqrt(sum(D_1.^2,1)), [size(D_1,1) 1]);
    D_1(isnan(D_1)) = 0;

    D_2 = D_2./repmat(sqrt(sum(D_2.^2,1)), [size(D_2,1) 1]);
    D_2(isnan(D_2)) = 0;

    lng_rate = lng_rate_0/(2^floor(gb_iter/10)); 

    %%% Error checking
    if (mod(inner_iter, 15) == 1)
        parfor bs = 1 : size(patch_list,3)

            % Forward Pass
            Level_1 = computeMultiScaleNextLayerPos(patch_list(:,:, bs), [5 5], [1 1], D_0, param);
            pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);
            Level_2 = computeMultiScaleNextLayerPos(pooling_1.Pooled, [5 5 size(D_0,2)], [1 1 1], D_1, param);
            pooling_2 = computePoolingLayerAvg_Fast_v2(max(Level_2.Layer, 0), 2);
            Level_3 = computeMultiScaleNextLayerPos(pooling_2.Pooled, [5 5 size(D_1,2)], [1 1 1], D_2, param);

            feats = [reshape(max(Level_3.Layer,0), [size(D_2,2) 1])];

            cL = zeros(10, 1);
            cL(currLabel(bs)+1, 1) = 1;

            %%% Softmax
            res_1 = [feats];
            p = exp( W'*res_1 )./repmat(sum(exp( W'*res_1 ), 1), [size(W,2) 1]);
            err2 = err2 + -sum(sum(cL.*log(p),1));
            [m, pr] = max(p);

            if (cL(pr, 1)) == 1
                classif_res2 = classif_res2 +1;
            end
        end
        disp('Corrected : ');
        disp(classif_res2/(size(patch_list,3)));
        disp(err2/(size(patch_list,3)));
    end
   
    if (mod(inner_iter, 5) == 0)
        save(sname, 'D_0', 'D_1', 'D_2', 'W', 'l_err', 'param', 'batchsize', 'nsamples', 'lng_rate', 'lng_rate_0', 'param', 'param2', 'param3');
    end

    it_counter = it_counter +1;
    l_err = [l_err classif_res/(size(patch_list,3))];
    end
end
display('End train');
%%
