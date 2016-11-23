%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example : Learning of a 3-layer architecture on Cifar-10
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
%% Parameters
%%% Layer 1
rfSize = 5; % Size of the patches (Layer 1)
% Dictionary 1
param.K=25;  % Number of atoms
param.lambda=0.1; % Regularization parameter for the l1-norm 
                  % (affects parsimony)

%%% Layer 2
rfSize2 = 5; % Size of the patches (Layer 2)
% Dictionary 2
param2.K=25;  
param2.lambda=0.1; 

%%% Layer 3
rfSize3 = 5; % Size of the patches (Layer 3)
% Dictionary 3
param3.K=50;  
param3.lambda=0.1; 

% Save file
sname = 'CIFAR_3layers_21-11_Learning_25-25-50_whitened_ppatch_v2.mat';
%% Train
train_images = zeros(30000, 3072);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Whitening
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numPatches = 400000;
CIFAR_DIM=[32 32 3];

% Extract random patches
patches = zeros(numPatches, rfSize*rfSize*3);
for i=1:numPatches
  if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end
  
  r = random('unid', CIFAR_DIM(1) - rfSize + 1);
  c = random('unid', CIFAR_DIM(2) - rfSize + 1);
  patch = reshape(train_images(mod(i-1,size(train_images,1))+1, :), CIFAR_DIM);
  patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
  patches(i,:) = patch(:)';
end

% Normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% Whiten
C = cov(patches);
M = mean(patches);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
clear patches;

%% Layer 1
poolSize = 2;
stride = 2;

nsamples = size(train_images,1);
train_images = reshape(train_images', [32 32 3, nsamples]);

patch_list = zeros(size(train_images));
for i = 1 : size(patch_list,4)
    patch_list(:,:,:, i) = permute(train_images(:,:,:, i), [2 1 3]);
end

clear train_images;

D_0 = [];
param.lambda2=0.01;
param.numThreads=-1; 
param.batchsize=400;
param.verbose=false;
param.mode = 2;
param.iter = 200;

spatch_list = [];
for i = 1 : 15 : size(patch_list,4);
    plist = im2colstep(patch_list(:,:,:, i), [rfSize rfSize 3], [1 1 1]);

    plist = bsxfun(@rdivide, bsxfun(@minus, plist', mean(plist,1)'), sqrt(var(plist,[],1)'+10));
    plist = (bsxfun(@minus, plist, M) * P)';
    
    spatch_list = [spatch_list plist];
end

spatch_list(:, sum(abs(spatch_list),1) == 0) = [];

spatch_list = (bsxfun(@minus, spatch_list', M) * P)';

disp('Learning init dictionary ..');
D_0 = mexTrainDL(spatch_list, param);
disp('Done..');

%% Layer 2

spatch_list = [];
for i = 1 : 15 : size(patch_list,4)

    Level_1 = computeMultiScaleNextLayerPosWhiten(patch_list(:,:,:, i), [rfSize rfSize 3], [1 1 1], D_0, M, P, param);
    pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);

    plist = im2colstep(pooling_1.Pooled, [rfSize2 rfSize2 size(D_0,2)], [1 1 1]);
    plist = plist./repmat(sqrt(sum(plist.^2,1)), [size(plist,1) 1]);
    plist(isnan(plist)) = 0;
    spatch_list = [spatch_list plist];
end

spatch_list(:, sum(abs(spatch_list),1) == 0) = [];

param2.lambda2=0.01;
param2.numThreads=-1; % number of threads
param2.batchsize=400;
param2.verbose=false;
param2.mode = 2;
param2.iter = 400;

disp('Learning init dictionary ..');
D_1 = mexTrainDL(spatch_list, param2);
disp('Done..');

clear spatch_list;

%% Layer 3

spatch_list = [];
for i = 1 : 15 : size(patch_list,4)
   
    Level_1 = computeMultiScaleNextLayerPosWhiten(patch_list(:,:,:, i), [rfSize rfSize 3], [1 1 1], D_0, M, P, param);
    pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);
    Level_2 = computeMultiScaleNextLayerPos(pooling_1.Pooled, [rfSize2 rfSize2 size(D_0,2)], [1 1 1], D_1, param2);
    pooling_2 = computePoolingLayerAvg_Fast_v2(max(Level_2.Layer, 0), 2);
    
    plist = im2colstep(pooling_2.Pooled, [rfSize3 rfSize3 size(D_1,2)], [1 1 1]);
    plist = plist./repmat(sqrt(sum(plist.^2,1)), [size(plist,1) 1]);
    plist(isnan(plist)) = 0;
    spatch_list = [spatch_list plist];
end

spatch_list(:, sum(abs(spatch_list),1) == 0) = [];

param3.lambda2=0.01;
param3.numThreads=-1; % number of threads
param3.batchsize=400;
param3.verbose=false;
param3.mode = 2;
param3.iter = 600;

disp('Learning init dictionary ..');
D_2 = mexTrainDL(spatch_list, param3);
disp('Done..');

clear spatch_list;

%%

W = randn( size(D_2,2), 10)/20;

save(sname, 'D_0', 'D_1', 'D_2', 'W', 'M', 'P');
%%
