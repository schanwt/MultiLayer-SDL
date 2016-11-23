function [ X ] = sparseCoding2( D , Y , T0)


ompparams = {'checkdict','off','messages',5};
ompfunc = @omp2;

% XtX = colnorms_squared(Y);
 XtX = normcols(Y);
G = D'*D;

X = ompfunc(D'*Y,XtX,G,T0,ompparams{:});


end
