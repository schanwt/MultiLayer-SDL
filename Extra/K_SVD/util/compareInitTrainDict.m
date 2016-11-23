function compareInitTrainDict( Dinit , Dr )

m = size(Dr,2);
tpatch = sqrt(size(Dr,1));
Dr = -Dr;
matches = matching(Dinit,Dr);

scrsz = get(0,'ScreenSize');
figure('Position',[1 scrsz(4)/2  scrsz(3) scrsz(4)/2])

cpt=1;

for i=1:m
    
    di = Dinit(:,matches(cpt,2));
    dt = Dr(:,matches(cpt,1));
    err = (di-dt).^2;
    
    errm = mean(err);
    
    di = reshape(di,tpatch,tpatch);
    dt = reshape(dt,tpatch,tpatch);
    err = reshape(err,tpatch,tpatch);
    
    subplot(3,m,cpt);
    imagesc(di);
    axis off;
    colormap(gray);
    subplot(3,m,cpt+m);
    imagesc(dt);
    colormap(gray)
    axis off;;
    subplot(3,m,cpt+2*m);
    imagesc(err);
    axis off;
    title(num2str(errm));
    
    cpt = cpt+1;
end


end

