function [ X ] = generateToysFromDict( D)
    

    %alpha = 0.1:0.1:0.9;
    
    alpha = 0.3:0.025:0.7;
    
    X = [];
    bnoise=0;
    noiseratio=20;

    if(bnoise==0)
        noiseg = zeros(size(D,1),1);
    else
        noiseg = randn(size(D,1),1)/noiseratio;
    end

    NumbofOrient = size(D,2);
    cpt=1;
    for i=1:NumbofOrient
        for j=i+4:NumbofOrient
            %cpt2=0;
            for k=1:length(alpha)
                XX = alpha(k) * D(:,j) + (1-alpha(k)) * D(:,i) + noiseg(:);
                X = [X XX];
                %cpt2 = cpt2 +1
            end
            cpt = cpt +1;
        end
    end
    
     p = randperm(size(X,2)); % matlab fct for random permutation
     
     X = X(:,p);

end

