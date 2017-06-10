classdef dataClass < handle
   
%     properties (SetAccess = private)
    properties
        X;
        scale;
        shift=0;
        atGPU;
        m;
        blobSize;
        cellInd;
    end
    
    
    methods
        function this = dataClass(fName,type,blobSize,scale,maxExamples,atGPU,varargin)
            % TODO: Make sure that blobSize is size 3.
            % varargin might include dataFormat,comp,shift
            numvarargs = length(varargin);
            if numvarargs > 3
                error('myclasses:dataClass:TooManyInputs', ...
                    'requires at most 3 optional inputs');
            end

            % set defaults 
            default_args = {'bin' 0 0};

            % now put these defaults into the valuesToUse cell array, 
            % and overwrite the ones specified in varargin.
            default_args(1:numvarargs) = varargin;
            % or ...
            % [optargs{1:numvarargs}] = varargin{:};

            % Place optional args in memorable variable names
            [dataFormat,comp,shift] = default_args{:};

            
            
            this.atGPU = atGPU;
            this.scale = scale;
            this.shift = shift;

            this.blobSize = blobSize;
            if strcmp(dataFormat,'bin')
                fid = fopen(fName,'rb');
                if comp
                    maxExamples = 2*maxExamples;
                end
                this.X = fread(fid,maxExamples*prod(this.blobSize),type);
                this.m = floor(length(this.X)/(prod(this.blobSize)));
                x_size = [this.blobSize , this.m];
                if comp
                    this.m = this.m/2;
                    x_size = [this.blobSize(1:end-1),2, this.blobSize(end), this.m];
                end
                this.X = reshape(this.X(1:(prod(x_size))),x_size);
                fclose(fid);
            end
            if strcmp(dataFormat,'mat')
                mat = load(fName);
                this.m = mat.m;
                this.X = mat.X;
                clear mat
            end


            
            if comp
                ind = cell(1,length(size(this.X)));
                for i=1:length(ind)
                    ind{i} = ':';
                end
                indreal = ind; indreal{4} = 1;
                indimag = ind;indimag{4} = 2;
                this.X = this.X(indreal{:})+1i*this.X(indimag{:});
                this.X = permute(this.X,[1,2,3,5,6,4]);
            end
        
            this.cellInd = cell(length(size(this.X)),1); 
            for i=1:(length(this.cellInd)-1)
                this.cellInd{i} = ':';
            end
        end
        
        function x = get(this,i)
            this.cellInd{end} = i;
            if this.atGPU
                x = gpuArray(single((this.X(this.cellInd{:})*this.scale)+this.shift));
            else
                x = single((this.X(this.cellInd{:})*this.scale)+this.shift);
            end
        end
        
    end
end
