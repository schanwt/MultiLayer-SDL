function outSize = predRowCol(iSize, s, stride)

    if numel(s) == 1
        s_1 = s;
        s_2 = s;
    else
        s_1 = s(1);
        s_2 = s(2);
    end

    if (length(stride) == 1)
        stride_1 = stride;
        stride_2 = stride;
    else
        stride_1 = stride(1);
        stride_2 = stride(2);
    end

    outSize(1) = floor( (iSize(1) - s_1)/stride_1 ) + 1 ;
    outSize(2) = floor( (iSize(2) - s_2)/stride_2 ) + 1 ;
end