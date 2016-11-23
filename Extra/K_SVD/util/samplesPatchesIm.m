    function [ p ,r] = samplesPatchesIm( OL , Nim , s ,seuil)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

%p = [];
%r = [];
p = zeros(s*s,Nim);
r = zeros(2,Nim); %[ 1 ... Nim_1 ]
                  %[ 2 ... Nim_2 ]
w = size(OL,1);
h = size(OL,2);

cpt=1;

while(cpt<=Nim)
    
    indx = randi([1,w-s+1],1);
    indy = randi([1,h-s+1],1);
    
    avance=1;
    ss = size(r,2); %Nim
    
    if(ss>0)
        Scale = zeros(2,ss); %(2, Nim)
        Scale(:,:) = s;
        

        v = [ indx indy s s ];
        a = [ r'  Scale' ];
        
             
        areas = rectint(a , v);
        maxint = max(areas);
        
        if(maxint>seuil)
            avance=0;
        end
        
    end
    
    if(avance==1)
        patch = OL(indx:indx+s-1,indy:indy+s-1);
        patch = patch(:);
        p(:,cpt) = patch;
        r(:,cpt) = [indx; indy];
        
        cpt = cpt+1;
    end
    
end

end

