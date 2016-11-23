%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example : Testing of a 3-layer architecture on Cifar-10
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath ../Extra/speedup
addpath ../Extra/spams-matlab/
addpath ../Extra/spams-matlab/build/
addpath ../Extra/K_SVD/
addpath ../Extra/K_SVD/util/
addpath ../Extra/Processing/
addpath ../Extra/liblinear-1.94/matlab/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ICI : Répertoire contenant la base d'images
addpath ./cifar-10-batches-mat/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Dictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('CIFAR_3layers_18-07_50-50-100_whitened_ppatch_v2.mat');

% Parameters (size of the patches on the 3 layers)
rfSize = 5;
rfSize2 = 5;
rfSize3 = 5;
%% Load the dataset
load('test_batch.mat');
test_images = data';
test_labels = labels;

test_images = reshape(test_images, [32 32 3, size(test_images,2)]);
test_images = permute(test_images, [2 1 3 4]);

%%
batchsize = 100;
nsamples = size(test_images,4);

full_te = zeros(size(W,1), size(test_images,4));
small_te =zeros(size(W,1), batchsize);


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

pass_max = floor(nsamples/batchsize);
it_counter = 1;
l_err= [];
for gb_iter = 1 : 1
    classif_res2 = 0;
    for inner_iter = 1 : pass_max
        classif_tmp = 0;
        idx = [(inner_iter-1)*batchsize+1 : inner_iter*batchsize];  
        patch_list = double(test_images(:,:,:,idx));
        currLabel = test_labels(idx);
        
        parfor bs = 1 : size(patch_list,4)

            % Forward pass
            Level_1 = computeMultiScaleNextLayerPosWhiten(patch_list(:,:,:, bs), [rfSize rfSize 3], [1 1 1], D_0, M, P, param);
            pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);
            Level_2 = computeMultiScaleNextLayerPos(pooling_1.Pooled, [rfSize2 rfSize2 size(D_0,2)], [1 1 1], D_1, param2);
            pooling_2 = computePoolingLayerAvg_Fast_v2(max(Level_2.Layer, 0), 2);
            Level_3 = computeMultiScaleNextLayerPos(pooling_2.Pooled, [rfSize3 rfSize3 size(D_1,2)], [1 1 1], D_2, param3);

            feats = [reshape(max(Level_3.Layer,0), [size(D_2,2) 1])];
            small_te(:,bs) = feats;
            
            cL = zeros(10, 1);
            cL(currLabel(bs)+1, 1) = 1;
            
            %%% Softmax
            res_1 = [feats];
            p = exp( W'*res_1 )./repmat(sum(exp( W'*res_1 ), 1), [size(W,2) 1]);
            [m, pr] = max(p);
            
            if (cL(pr, 1)) == 1
                classif_res2 = classif_res2 +1;
                classif_tmp = classif_tmp + 1;
            end
            
        end
        disp(classif_tmp/size(patch_list,4));
        full_te(:, idx) = small_te;
        
    end
    disp('Results on "Test" set: ');
    disp(classif_res2/(nsamples));
end

display('End Test');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This part is used to compare with the results of a SVM

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
load('data_batch_4.mat');
train_images(30001:40000, :) = data;
train_labels (30001:40000) = labels;
load('data_batch_5.mat');
train_images(40001:50000, :) = data;
train_labels (40001:50000) = labels;

nsamples = size(train_images,1);

train_images = reshape(train_images', [32 32 3, nsamples]);
train_images = permute(train_images, [2 1 3 4]);

test_images = train_images;
test_labels = train_labels;

full_tr = zeros(size(W,1), size(test_images,4));
small_tr =zeros(size(W,1), batchsize);

pass_max = floor(nsamples/batchsize);
it_counter = 1;
l_err= [];
for gb_iter = 1 : 1
    classif_res2 = 0;
    for inner_iter = 1 : pass_max
        classif_tmp = 0;
        idx = [(inner_iter-1)*batchsize+1 : inner_iter*batchsize];
        
        patch_list = double(test_images(:,:,:,idx));
        currLabel = test_labels(idx);

        %%% Error checking
        parfor bs = 1 : size(patch_list,4)
            
            % Forward pass
            Level_1 = computeMultiScaleNextLayerPosWhiten(patch_list(:,:,:, bs), [rfSize rfSize 3], [1 1 1], D_0, M, P, param);
            pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);
            Level_2 = computeMultiScaleNextLayerPos(pooling_1.Pooled, [rfSize2 rfSize2 size(D_0,2)], [1 1 1], D_1, param2);
            pooling_2 = computePoolingLayerAvg_Fast_v2(max(Level_2.Layer, 0), 2);
            Level_3 = computeMultiScaleNextLayerPos(pooling_2.Pooled, [rfSize3 rfSize3 size(D_1,2)], [1 1 1], D_2, param3);

            feats = [reshape(max(Level_3.Layer,0), [size(D_2,2) 1])];
            small_tr(:,bs) = feats;
            
            cL = zeros(10, 1);
            cL(currLabel(bs)+1, 1) = 1;
            
            %%% Softmax
            res_1 = [feats];
            p = exp( W'*res_1 )./repmat(sum(exp( W'*res_1 ), 1), [size(W,2) 1]);
            [m, pr] = max(p);
            
            if (cL(pr, 1)) == 1
                classif_res2 = classif_res2 +1;
                classif_tmp = classif_tmp + 1;
            end            
        end
        full_tr(:, idx) = small_tr;     
    end
    disp('Results on "Training" set: ');
    disp(classif_res2/(nsamples));
end

display('End train');

%%
display('Starting SVM');

load('test_batch.mat');
test_images = data';
test_labels = labels;

opt = strcat('-s 6 -q -c',32,num2str(10),' -p',32, num2str(1)); % -s 5

model = train(double(train_labels), sparse( full_tr )', opt);
[predicted_label, acc, estimates] = predict(double(test_labels), sparse( full_te )', model , '-b 1');

%%


display('Starting SVM');


full_tr = full_tr';
full_te = full_te';

trainXC_mean = mean(full_tr);
trainXC_sd = sqrt(var(full_tr)+0.01);

full_tr = bsxfun(@rdivide, bsxfun(@minus, full_tr, trainXC_mean), trainXC_sd);
full_tr = [full_tr, ones(size(full_tr,1),1)];

full_te = bsxfun(@rdivide, bsxfun(@minus, full_te, trainXC_mean), trainXC_sd);
full_te = [full_te, ones(size(full_te,1),1)];

opt = strcat('-s 6 -q -c',32,num2str(100),' -p',32, num2str(1)); % -s 5
% model = train(double(train_labels), sparse( (full_tr - repmat(mean(full_tr,2), [1 size(full_tr,2)])).*(full_tr~=0)  ), opt);
% [predicted_label, acc, estimates] = predict(double(test_labels), sparse( (full_te - repmat(mean(full_tr,2), [1 size(full_te,2)])).*(full_te~=0)  ), model , '-b 1');

disp('Results on "Test" set with linear SVM: ');
model = train(double(train_labels), sparse( full_tr ), opt);
[predicted_label, acc, estimates] = predict(double(test_labels), sparse( full_te ), model , '-b 1');

save('CIFAR_3layers_whitened_ppatch_v2_LAST_TEST.mat', 'D_0', 'D_1', 'D_2', 'W', 'M', 'P', 'l_err', 'param', 'param2', 'param3', 'batchsize', 'nsamples');

clear all;


