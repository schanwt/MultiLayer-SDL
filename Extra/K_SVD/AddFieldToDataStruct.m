function AddFieldToDataStruct(processList)
% Add a field to the DataBox struct
    for list_index = 1 : length(processList)
        display(processList(list_index).name);
        load(processList(list_index).name);

        % DO some stuffs
        % DataStruct
        n = DataStruct(1).video;
        tmp = regexp(n,'\','split');
        %display(tmp);
        lstr = [];
        for i = 4 : 7
           lstr = strcat(lstr, '/', tmp{i});
           
        end
        lstr = strcat('/nethome/chanwast/.windows',lstr);
        %display(lstr);
        
        DataStruct(1).servervideo = lstr;
        save(processList(list_index).name, 'DataStruct');
    end

end




