function ims = convert_gt(D,output_fid,output_data_type)  
    s = size(D.X);
    num_batches = s(end);

    ind = cell(1,length(D.blobSize));
    for i=1:length(ind)
        ind{i} = ':';
    end

    for i=1:num_batches
            tic;
            A = zeros(2,D.blobSize(4)*100);
            ims = D.get(i); %data is in (0,1)
            % read image
            for j=1:D.blobSize(4)
                ind{end} = j;
                im = ims(ind{:});
                ca = mat2cell(im(:,:,1), D.blobSize(1)/10*ones(1,10), D.blobSize(2)/10*ones(1,10));
                labels = zeros(2,100);
                for k=1:100
                    if(sum(sum(ca{k}))>10)
                        labels(1,k)=1;
                    else
                        labels(2,k)=1;
                    end
                end
                %Plug in
                A(:,100*(j-1)+1:100*j) = labels;
            end
            fprintf('%d\n',i);
            toc
            fwrite(output_fid,A,output_data_type); 
    end
    end

    




        
    