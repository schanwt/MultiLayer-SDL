function show3ddictionary(D, Dsize)
% D assumed to be in columns size dxK

[d, K] = size(D);

s1 = Dsize(1);
s2 = Dsize(2);
z = Dsize(3);

% D1 = zeros(s1*s2,K);
% D2 = zeros(s1*s2,K);
% D3 = zeros(s1*s2,K);

% for i=1:K
%    D(:,i) = imnormalize(D(:,i));
%    D1(:,i) = D(1:s1*s2,i);
%    D2(:,i) = D(s1*s2+1:2*s1*s2,i);
%    D3(:,i) = D(2*s1*s2+1:end,i);
% end

for i=1:K
   D(:,i) = imnormalize(D(:,i));
   for j = 1 : z
   D_r(:,i,j) = D(1+(j-1)*s1*s2:j*s1*s2,i);
   end
end

%t = sqrt(d);
% doit etre entier car patchs carrés
%Dr = reshape(D,t,t,K);

%drawPatches(Dr);
%subplot(3,1,1);
% drawPatches(D1, [s1 s2],1);
% %subplot(3,1,2);
% drawPatches(D2, [64 32],2);
% %subplot(3,1,3);
% drawPatches(D3, [64 32],3);
for j = 1 : z
    drawPatches(D_r(:,:,j), [s1 s2]);
end


end