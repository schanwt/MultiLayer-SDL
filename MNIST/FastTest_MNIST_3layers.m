%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example : Testing of a 3-layer architecture on MNIST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath ../Extra/speedup
addpath ../Extra/spams-matlab/
addpath ../Extra/spams-matlab/build/
addpath ../Extra/K_SVD/
addpath ../Extra/K_SVD/util/
addpath ../Extra/Processing/
addpath ../Extra/liblinear-1.94/matlab/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here : Path to the dataset
addpath ./Dataset/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train

% train_images = loadMNISTImages('train-images.idx3-ubyte');
% train_labels = loadMNISTLabels('train-labels.idx1-ubyte');

load('MNIST_100_perClass_1.mat', 'train_images', 'train_labels');

test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the dictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('MNIST_3layers_23-11_25-25-50_1000perClass.mat');

% Parameters
rfSize = 5;
rfSize2 = 5;
rfSize3 = 5;

batchsize = 100;
nsamples = size(test_images,2);

patch_list_tmp = double(test_images);
patch_list_tmp = reshape(patch_list_tmp, [28 28 size(test_images,2)]);
patch_list_tmp = imresize(patch_list_tmp, [32 32]);

full_te = zeros(size(W,1), size(test_images,2));
small_te =zeros(size(W,1), size(batchsize,2));

clear param;
param.lambda=0.1; % 0.05
param.lambda2=0.01;
param.mode = 2;

pass_max = floor(nsamples/batchsize);
it_counter = 1;
l_err= [];
for gb_iter = 1 : 1
    classif_res2 = 0;
    for inner_iter = 1 : pass_max
        classif_tmp = 0;
        idx = [(inner_iter-1)*batchsize+1 : inner_iter*batchsize];
        
        patch_list = patch_list_tmp(:,:,idx);
        currLabel = test_labels(idx);
            
        %%% Error checking
        parfor bs = 1 : size(patch_list,3)
            
            % Forward Pass
            Level_1 = computeMultiScaleNextLayerPos(patch_list(:,:, bs), [rfSize rfSize], [1 1], D_0, param);
            pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);
            Level_2 = computeMultiScaleNextLayerPos(pooling_1.Pooled, [rfSize2 rfSize2 size(D_0,2)], [1 1 1], D_1, param);
            pooling_2 = computePoolingLayerAvg_Fast_v2(max(Level_2.Layer, 0), 2);
            Level_3 = computeMultiScaleNextLayerPos(pooling_2.Pooled, [rfSize3 rfSize3 size(D_1,2)], [1 1 1], D_2, param);

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
        disp(classif_tmp/size(patch_list,3));
        full_te(:, idx) = small_te;       
    end
    disp('Results : ');
    disp(classif_res2/(nsamples));
end

display('End test');

%% Full (Comparison with a SVM)

test_images = train_images;
test_labels = train_labels;

batchsize = 100;

nsamples = size(test_images,2);

patch_list_tmp = double(test_images);
patch_list_tmp = reshape(patch_list_tmp, [28 28 size(test_images,2)]);
patch_list_tmp = imresize(patch_list_tmp, [32 32]);

full_tr = zeros(size(W,1), size(test_images,2));
small_tr =zeros(size(W,1), size(batchsize,2));

clear param;
param.lambda=0.1; % 0.05
param.lambda2=0.01;
param.mode = 2;

pass_max = floor(nsamples/batchsize);
it_counter = 1;
l_err= [];
for gb_iter = 1 : 1
    classif_res2 = 0;
    for inner_iter = 1 : pass_max
        classif_tmp = 0;
        idx = [(inner_iter-1)*batchsize+1 : inner_iter*batchsize];
        
        patch_list = patch_list_tmp(:,:,idx);
        currLabel = test_labels(idx);
          
        %%% Error checking
        parfor bs = 1 : size(patch_list,3)
            id = (inner_iter-1)*batchsize+bs;
            
            % Forward Pass
            Level_1 = computeMultiScaleNextLayerPos(patch_list(:,:, bs), [5 5], [1 1], D_0, param);
            pooling_1 = computePoolingLayerAvg_Fast_v2(max(Level_1.Layer, 0), 2);
            Level_2 = computeMultiScaleNextLayerPos(pooling_1.Pooled, [5 5 size(D_0,2)], [1 1 1], D_1, param);
            pooling_2 = computePoolingLayerAvg_Fast_v2(max(Level_2.Layer, 0), 2);
            Level_3 = computeMultiScaleNextLayerPos(pooling_2.Pooled, [5 5 size(D_1,2)], [1 1 1], D_2, param);

            %feats = [feats; reshape(max(Level_5.Layer,0), [size(D_4,2) 1])];
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
    disp('Results : ');
    disp(classif_res2/(nsamples));
end

display('End train');
%%

display('Starting SVM');

test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

opt = strcat('-s 6 -q -c',32,num2str(1000),' -p',32, num2str(0.1)); % -s 5
model = train(double(train_labels), sparse(full_tr'), opt);

[predicted_label, acc, estimates] = predict(double(test_labels), sparse(full_te'), model , '-b 1');

clear all;
