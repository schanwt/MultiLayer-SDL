%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here : Path to the images (dataset)
addpath ./Dataset/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train

train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');

%% Parameters

nbImagesPerClass = 100; % x10 (nb of class) for the size of the training set
sname = 'MNIST_100_perClass_1.mat';


strClassList = struct();

for i = 1 : 10
    strClassList(1).Class(i).Images = train_images(:, train_labels == i-1);
end

idx = randperm(5000);

tmpIm = zeros( size(train_images,1), nbImagesPerClass*10);
tmpLabel = zeros(nbImagesPerClass*10,1);

for i = 1 : 10
    tmpIm(:, (i-1)*nbImagesPerClass+1 : i*nbImagesPerClass) = strClassList(1).Class(i).Images(:, idx(1:nbImagesPerClass));
    tmpLabel((i-1)*nbImagesPerClass+1 : i*nbImagesPerClass) = i-1;
end

idx = randperm(nbImagesPerClass*10);

clear train_images;
clear train_labels;

train_images = tmpIm(:, idx);
train_labels = tmpLabel(idx);

save(sname, 'train_images', 'train_labels');
%%


