function [d_k, x_k, w_k] = optimize_dk(k, Y, D, X)

w_k = find(abs(X(k,:))~=0);
%disp(size(Y));
%disp(length(w_k));
if length(w_k)>0
    Ek = Y - D*X + D(:,k)*X(k,:);
    Omega = zeros(size(Ek,2), length(w_k));
    for i = 1 : length(w_k)
       Omega(w_k(i),i) = 1; 
    end
    Ekr = Ek*Omega;
    [U, S, V] = svds(Ekr,1);
    d_k = U(:,1);
    x_k = V(:,1)*S;
    
else

%    w_k = ceil(size(Y,2)*rand());
%    x_k = 1;
%    %disp(w_k);
%    d_k = Y(:, w_k);

   w_k = ceil(size(Y,2)*rand());
   x_k = 1;
   %disp(w_k);
   d_k = D(:,k);
end

end