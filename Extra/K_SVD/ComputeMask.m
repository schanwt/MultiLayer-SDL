function mask = ComputeMask(patch, motions, contours)

mask = zeros(size(patch,1),size(patch,2));
if motions == 1
    mask = abs(patch(:,:,1)-patch(:,:,2))>15 | abs(patch(:,:,2)-patch(:,:,3))>15;
elseif contours == 1
    [FX,FY] = gradient(patch(:,:,1));
    mask = sqrt(FX.^2+FY.^2)>20;
else
    mask = ones(size(patch,1),size(patch,2));
end