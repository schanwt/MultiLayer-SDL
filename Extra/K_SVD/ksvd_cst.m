function [D] = ksvd_cst(Y, initdict, iternum, T0, ind_groups)
    err = [];
    D = initdict;

    % parameter of the optimization procedure are chosen
    param.L=T0; % not more than 10 non-zeros coefficients
    param.eps=0.1; % squared norm of the residual should be less than 0.1
    param.numThreads=-1; % number of processors/cores to use; the default choice is -1
    % and uses all the cores of the machine
    
    batchsize = 200;
    
    for i = 1 : iternum
        disp(i);
        
        rd = randperm(size(Y,2));
        newSet = Y(:, rd(1:batchsize));

        
        X=mexSOMP(newSet, D, int32([0]), param);
        %X=mexSOMP(Y, D, int32(ind_groups), param);
        for k = 1 : size(D, 2) - 1
           [d_k, x_k, w_k] = optimize_dk(k, newSet, D, X);
           D(:,k) = d_k;
           X(k, w_k) = x_k;
        end
        %err = [err sum(sum((Y-D*X).^2))];
        %disp(sum(sum((Y-D*X).^2)));
        disp(sum(sum((newSet-D*X).^2)));
    end
    

end