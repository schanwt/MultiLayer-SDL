%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example : Learning (init) of a 3-layer architecture on MNIST
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

load('MNIST_100_perClass_1.mat', 'train_images', 'train_labels');

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

% Save name
sname = 'MNIST_3layers_22-11_Learning_25-25-50_100perClass.mat';

%% Layer 1

poolSize = 2;
stride = 2;

nsamples = size(train_images,2);

D_0 = [];

patch_list_tmp = double(train_images(:,1:nsamples));

patch_list_tmp = reshape(patch_list_tmp, [28 28 nsamples]);
patch_list = imresize(patch_list_tmp, [32 32]);


spatch_list = [];
for i = 1 : 15 : size(patch_list,3)
	tmp = double(patch_list(:,:,i));
    plist = im2colstep(tmp, [rfSize rfSize], [1 1]);
    plist = plist./repmat(sqrt(sum(plist.^2,1)), [size(plist,1) 1]);
    plist(isnan(plist)) = 0;
    spatch_list = [spatch_list plist];
end

spatch_list(:, sum(abs(spatch_list),1) == 0) = [];

param.lambda2=0.01;
param.numThreads=-1; 
param.batchsize=400;
param.verbose=false;
param.mode = 2;
param.iter = 200;

disp('Learning init dictionary ..');
D_0 = mexTrainDL(spatch_list, param);
disp('Done..');

%% Layer 2

clear param;
param.lambda=0.1; % 0.05
param.lambda2=0.01;
param.mode = 2;

spatch_list = [];
for i = 1 : 15 : size(patch_list,3)
   
   % Layer 1
    Level_1 = computeMultiScaleNextLayerPos(patch_list(:,:, i), [rfSize rfSize], [1 1], D_0, param);
    pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);

    plist = im2colstep(pooling_1.Pooled, [rfSize2 rfSize2 size(D_0,2)], [1 1 1]);
    plist = plist./repmat(sqrt(sum(plist.^2,1)), [size(plist,1) 1]);
    plist(isnan(plist)) = 0;
    spatch_list = [spatch_list plist];
end

spatch_list(:, sum(abs(spatch_list),1) == 0) = [];

param2.lambda2=0.01;
param2.numThreads=-1; 
param2.batchsize=400;
param2.verbose=false;
param2.mode = 2;
param2.iter = 200;

disp('Learning init dictionary ..');
D_1 = mexTrainDL(spatch_list, param2);
disp('Done..');

clear spatch_list;

%% Layer 3

spatch_list = [];
for i = 1 : 15 : size(patch_list,3)
   
   % Layer 1
    Level_1 = computeMultiScaleNextLayerPos(patch_list(:,:, i), [rfSize rfSize], [1 1], D_0, param);
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
param3.numThreads=-1; 
param3.batchsize=400;
param3.verbose=false;
param3.mode = 2;
param3.iter = 200;

disp('Learning init dictionary ..');
D_2 = mexTrainDL(spatch_list, param3);
disp('Done..');

clear spatch_list;

%%

W = randn( size(D_2,2), 10)/20;

save(sname, 'D_0', 'D_1', 'D_2', 'W');


