classdef lossClass < handle
   
    properties (SetAccess = private)
        type;
    end
    
    
    methods
        function this = lossClass(lossType)
            
            switch lossType
                case 'MCLogLoss'
                    this.type = 1;
                case 'SquaredLoss'
                    this.type = 2;
                case 'BinLogLoss'
                    this.type = 3;
                case 'HeatMapLogLoss'
                    this.type = 4;
                case 'HeatMapLogLossClipped'
                    this.type = 5;
                case 'MCHingeLoss'
                    this.type = 6;
                otherwise
                    assert(false,'Unknown loss type')
            end
            
        end
        
        function loss=LossAndErr(this,pred,Y)
            switch this.type
                case 1 % multiclass logistic loss
                    [k,m] = size(pred);
                    [loss,ind]=max(pred);
                    pred = pred-repmat(loss,k,1);
                    [~,y]=max(Y);
                    err = sum(ind~=y)/m;
                    
                    valY = sum(pred.*Y);
                    loss = pred - repmat(valY,k,1);
                    loss = sum(logsumexp(loss))/m;
                    loss = [loss;err];
                case 2 % Squared Loss
                    loss = 0.5*mean(sum(abs(pred-Y).^2));
                    loss = sum(loss)/numel(loss);
                case 3 % binary log loss
                    loss = -pred.*Y;
                    err = mean(sum(loss>=0));
                    loss(loss>0) = loss(loss>0) + log(1+exp(-loss(loss>0)));
                    loss(loss<=0) = log(1+exp(loss(loss<=0)));
                    loss = [mean(loss) ; err];
                
                case 4 %heat map log loss
                    s = size(pred);
                    l = length(s);
                    pred = reshape(pred,prod(s(1:l-1)),s(l));
                    Y = reshape(Y,prod(s(1:l-1)),s(l));
                    bg_softmax = zeros(1,size(pred,2));
                    fg = zeros(1,size(pred,2));
                    for i=1:size(pred,2)
                        y = Y(:,i);
                        p = pred(:,i);
                        bg_softmax(i) = logsumexp(p(~y))*sum(y);
                        fg(i) = sum(y);
                    end
                    loss = sum(-pred.*Y) + bg_softmax;
                    loss = loss./fg;
                    loss = sum(loss)/numel(loss);
                    
                 case 5 %heat map log loss clipped (v2)
                    s = size(pred);
                    l = length(s);
                    pred = reshape(pred,prod(s(1:l-1)),s(l));
                    Y = reshape(Y,prod(s(1:l-1)),s(l));
                    loss = (zeros(1,size(pred,2)));
                    if isa(pred,'gpuArray') %TODO: like in the lossWrapper
                        loss = gpuArray(loss);
                    end
                    for i=1:size(pred,2)
                        y = logical(real(Y(:,i)));
                        p = pred(:,i);
                        bg_softmax = logsumexp(p(~y));
                        diffs = log(1+exp(bg_softmax-p(y)));
                        loss(i) = (real(sum(diffs)/numel(diffs)));
                    end
                    loss = sum(loss)/numel(loss);
                case 6
                    [k,m] = size(pred);
                    [loss,ind]=max(pred);
%                     pred = pred-repmat(loss,k,1);
                    [~,y]=max(Y);
                    Y = gather(Y);
                    err = sum(ind~=y)/m;
                    
                    valY = sum(pred.*Y);
                    loss = pred - repmat(valY,k,1);
                    not_indicator = cell2mat(arrayfun(@(i) double(1:k~=Y(i))',...
                                    1:m,'UniformOutput',0));
                    loss = not_indicator + loss;
                    loss = sum(logsumexp(loss))/m;
                    loss = [loss;err];
                    
            end
        end

        function delta=Grad(this,pred,Y)
            switch this.type

                case 1 % multiclass logistic loss
                    bla = pred-repmat(max(pred),size(pred,1),1);
                    bla = exp(bla);
                    bla=bla./repmat(sum(bla),size(bla,1),1);
                    delta = (bla - Y)/size(bla,2);
                    
                case 2 % SquaredLoss
                    delta = (pred-Y)/size(Y,2);
                    delta = delta./size(pred,4);

                case 3 % binary log loss
                    delta = -Y./(1+exp(pred.*Y))/size(Y,2);
                
                case 4 %heat map log loss
                    bg_softmax = zeros(1,1,1,size(pred,4));
                    fg_count = zeros(1,1,1,size(pred,4));
                    for i=1:size(pred,4)
                        y = Y(:,:,:,i);
                        p = pred(:,:,:,i);
                        fg_count(i) = sum(y(:));
                        bg_softmax(i) = fg_count(i)/sum(exp(p(~y(:))));

                    end    
                    delta = bsxfun(@mtimes,exp(pred), bg_softmax);
                    delta(logical(Y)) = -1;
                    delta = bsxfun(@rdivide,delta,fg_count);
                    delta = delta./size(pred,4);
                case 5 %heat map log loss clipped (v2)
                    delta = pred-pred;                   
                    for i=1:size(pred,4)
                        y = logical(real(Y(:,:,:,i)));
                        p = pred(:,:,:,i);
                        d = p-p;
                        bg_softmax = logsumexp(p(~y(:)));
                        d(y) = -1./(sum(y(:))*(1+exp(-(bg_softmax-p(y)))));
                        val = -sum(d(:))/sum(exp(p(~y(:))));
                        d(~y) = exp(p(~y))*val;
                        delta(:,:,:,i) = d;
                    end 
                    delta = delta./size(pred,4);
                case 6 % multiclass hinge loss
                    [k,m] = size(pred);
                    [loss,ind]=max(pred);
%                     pred = pred-repmat(loss,k,1);
                    [~,y]=max(Y);
                    Y = gather(Y);
                    err = sum(ind~=y)/m;
                    
                    valY = sum(pred.*Y);
                    loss = pred - repmat(valY,k,1);
                    not_indicator = cell2mat(arrayfun(@(i) double(1:k~=Y(i))',...
                                    1:m,'UniformOutput',0));
                    loss = not_indicator + loss;
                    bla = exp(loss);
                    delta=bla./repmat(sum(bla),size(bla,1),1);
                    delta = (delta-Y)/size(bla,2);
                    
            end
            
        end
        
        
    end
end
