function drawPatches( patcheso, pSize, varargin)

s2 = size(patcheso,1);
Nt =  size(patcheso,2);
s_1 = pSize(1);
s_2 = pSize(2);
patches = reshape(patcheso ,s_1,s_2,Nt );


% nx = ceil(sqrt(Nt));
ny= ceil(sqrt(Nt));
nx = ceil(Nt/ny);


cpt = 1;
% ppatches = zeros(s*nx,s*ny); % /!\
% for i=1:nx
%     for j=1:ny
%         xdeb = (i-1)*s+1;
%         xfin = (i)*s;
%         ydeb = (j-1)*s+1;
%         yfin = (j)*s;
%         if(cpt <= size(patches,3))
%             ppatches(xdeb:xfin,ydeb:yfin) = patches(:,:,cpt);
%         end
%         cpt = cpt+1;
%     end
% end

ppatches = zeros(s_1*nx+nx+1,s_2*ny+ny+1); % /!\
ppatches(1,:) = 0.7;
ppatches(:,1) = 0.7;
for i=1:nx
    for j=1:ny
        xdeb = (i-1)*s_1+1+i;
        xfin = (i)*s_1+i;
        ydeb = (j-1)*s_2+1+j;
        yfin = (j)*s_2+j;
        if(cpt <= size(patches,3))
            ppatches(xdeb:xfin,ydeb:yfin) = patches(:,:,cpt);
            ppatches(xfin+1,:) = 0.7;
            ppatches(:,yfin+1) = 0.7;
        end
        cpt = cpt+1;
    end
end


if length(varargin) > 0
    figure(varargin{1}(1));
else
    figure;
end
colormap(gray);
imagesc(ppatches);


end