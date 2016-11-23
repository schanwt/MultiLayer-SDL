function [ matches ] = matching( D,Dtrain )

    matSim = Dtrain' * D;

    figure();
    imagesc(matSim);
    colorbar();

    m = size(D,2);
    matches = zeros(m,2);

    cpt =1;
    while(cpt<=m)
        maxi = max(max(matSim));
        [matches(cpt,1),matches(cpt,2)] = find(matSim==maxi);
        matSim(matches(cpt,1),:) = -10;
        matSim(:,matches(cpt,2)) = -10;
        
        cpt=cpt+1;
    end
    
    matches =  sortrows(matches,2);


end

