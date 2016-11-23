function spatch_list = ExtractPatchSet(patch, s)

l = size(patch,1);
l2 = size(patch,2);
dis = floor(s/2);

spatch_list = [];

for i = dis+1 : l-dis
    for j = dis+1 : l2-dis
        %if (mask(i,j)==1)
            subpatch = patch(i-dis : i+dis, j-dis : j+dis, :);
            col_subpatch = reshape(subpatch, (2*dis+1).^2*size(patch,3), 1);
            col_msubpatch = mean(col_subpatch);
            col_subpatch = (col_subpatch - col_msubpatch)/sqrt(sum((col_subpatch - col_msubpatch).^2+0.001));
            spatch_list = [spatch_list col_subpatch];
        %end
    end
end