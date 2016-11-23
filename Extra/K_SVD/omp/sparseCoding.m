
function [ X ] = sparseCoding(data,D,T0)

% Sparse coding solving the optimization problem
%
%       min  |X - D*GAMMA|_2     s.t.  |GAMMA|_0 <= T
%      gamma
%
G = D'*D;

ompparams = {'checkdict','off','messages',-1};
ompfunc = @omp;

X = ompfunc(D'*data,G,T0,ompparams{:});


end
