function [D, X, err] = ksvd(Y, initdict, iternum, T0)
    err = [];
    D = initdict;
    for i = 1 : iternum
        disp(i);
        X = sparseCoding(Y, D, T0);
        for k = 1 : size(D, 2)
           [d_k, x_k, w_k] = optimize_dk(k, Y, D, X);
           D(:,k) = d_k;
           X(k, w_k) = x_k;
        end
        err = [err sum(sum((Y-D*X).^2))];
    end
    

end