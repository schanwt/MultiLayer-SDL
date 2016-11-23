%% Test for the SPAMS toolbox
%%

addpath ../ExternalCode/spams-matlab/build/
addpath ../ExternalCode/spams-matlab/
%%
tic;

clear processName;  
processName(1).name = 'Dataset/GT/Dataset_Diving-Side*';
processName(2).name = 'Dataset/GT/Dataset_Golf-Swing-Back*';
processName(3).name = 'Dataset/GT/Dataset_Golf-Swing-Front*';
processName(4).name = 'Dataset/GT/Dataset_Golf-Swing-Side*';
processName(5).name = 'Dataset/GT/Dataset_Kicking-Front*';
processName(6).name = 'Dataset/GT/Dataset_Kicking-Side*';
processName(7).name = 'Dataset/GT/Dataset_Riding-Horse*';
processName(8).name = 'Dataset/GT/Dataset_Run-Side*';
processName(9).name = 'Dataset/GT/Dataset_SkateBoarding-Front*';
processName(10).name = 'Dataset/GT/Dataset_Swing-Bench*';
processName(11).name = 'Dataset/GT/Dataset_Swing-SideAngle*';
processName(12).name = 'Dataset/GT/Dataset_Walk-Front*';
% Note : Lifting is removed due to no ground truth (former N°7 -> Riding N°8)


spatch = [];
spatchlist = [];

for i = 1 : length(processName)
    processList = dir(processName(i).name);
    [spatchlist, col_subpatch ] = UCF_ExtractFeatures(processList(1), 5);
    spatch = [spatch spatchlist];
end
toc;

%%

param.K = 150;
param.lambda = 1;
param.iter = 200;
param.mode = 3;
param.modeD = 0;    

[D] = mexTrainDL(spatchlist,param);

%%
%% Dictionary reduction

Idx = [];

D = Dbest_UCF;

for i = 1 : size(D,2)
    col = D(:,i);
    res = sum(col(1:25) - col(26:50)).^2 + sum(col(26:50) - col(51:75)).^2; 
    if res > 0.2
       Idx(end+1) = i; 
    end
end
