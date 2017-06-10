function [fiel] = createCSV(results,filename)
% assumption - results holds structs with the same structure.
f = fields(results{1});
fiel = {};
l = 0;

for i=1:length(f)
    cur_l = numel(results{1}.(f{i}));
    if cur_l<=10 &&(~strcmp('function_handle',class(results{1}.(f{i}))))
        fiel{end+1} = f{i};
        l = l+cur_l;
    end
end
        
M = zeros(l,length(results));

for i=1:length(results)
res = results{i};
col = [];
for j=1:length(fiel)
    cur = gather(res.(fiel{j}))';
    col = [col;cur(:)];
end
M(:,i) = col;
end
csvwrite(filename,M');