function ims = convert_input_fixed(D,output_fid,output_data_type)  
    s = size(D.X);
    num_batches = s(end);

    ind = cell(1,length(D.blobSize));
    for i=1:length(ind)
        ind{i} = ':';
    end

    for i=1:num_batches
            tic;
            A = zeros(D.blobSize(1)/10,D.blobSize(2)/10,2,D.blobSize(4)*100);
            ims = D.get(i); %data is in (0,1)
            % read image
            for j=1:D.blobSize(4)
                ind{end} = j;
                im = ims(ind{:});
                % compute and save gradient
                if size(im,3)==3
                    im = rgb2gray(im);
                end
                [dX,dY] = imgradientxy(im);
                % rescale from (-4,4) to (0,256)
                dX = (dX+4)*32;
%                 dY = dY;
                dY = (dY+4)*32;
                grad = dX+1i*dY;
                ca = mat2cell(grad, D.blobSize(1)/10*ones(1,10), D.blobSize(2)/10*ones(1,10));
                grads = zeros(D.blobSize(1)/10,D.blobSize(2)/10,100);
                for k=1:100
                    grads(:,:,k) = ca{k};
                end
                %Plug in
                A(:,:,1,100*(j-1)+1:100*j) = real(grads);
                A(:,:,2,100*(j-1)+1:100*j) = imag(grads);
            end

            fprintf('%d\n',i);
            toc
            fwrite(output_fid,A,output_data_type); 
    end
    end

    




        
    