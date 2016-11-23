function   showdictionary( D, pSize)
% D assumed to be in columns size dxK

[d,K] = size(D);

for i=1:K
   D(:,i) = imnormalize(D(:,i)); 
end

%t = sqrt(d);
% doit etre entier car patchs carrés
%Dr = reshape(D,t,t,K);

%drawPatches(Dr);

drawPatches(D, pSize);


end

