classdef ConvNet < handle
    properties
        net;
        testNet;
    end
    properties (SetAccess = private)
       
        theta;
        dtheta
        nLayers;
        O;
        delta;
        lossInd;
        
    end
    properties (SetAccess = private, GetAccess = private)
        mygpuArray;
        mygather;
    end 
    
    
    methods
        
        function this = ConvNet(netD,atGPU)
            % Constructor. See demoMNIST and demoDAG for examples
            if atGPU
                this.mygpuArray = @(x) gpuArray(x);
                this.mygather = @(x) gather(x);
            else
                this.mygpuArray = @(x) x;
                this.mygather = @(x) x;
            end
            this.initializeNet(netD,atGPU);
            this.nLayers = length(this.net);
            this.delta = cell(size(this.O));
            this.testNet = [];
        end
        
        function setTestNet(this,testNet)
            this.testNet=testNet;
        end
        
        % forward function
        function forward(this,I)
            
            for i=1:this.nLayers
                outInd = this.net{i}.outInd; 
                inInd = this.net{i}.inInd; 
                switch this.net{i}.type
                    case 'input'
                        this.O{outInd} = this.net{i}.data.get(I);
                    case 'duplicate'
                        for j=outInd
                            this.O{j} = this.O{inInd};
                        end
                    case 'im2col'
                        this.O{outInd} = this.O{inInd}(this.net{i}.im2colInd); % fast im2col
                    case 'max'
                        this.O{outInd} = max(this.O{inInd});
                    case 'mean'
                        this.O{outInd} = mean(this.O{inInd});
                    case 'reshape'
                        this.O{outInd} = reshape(this.O{inInd},this.net{i}.newshape);
                    case 'permute'
                        this.O{outInd} = permute(this.O{inInd},this.net{i}.newshape);
                    case 'affine'
                        this.O{outInd} = reshape(this.theta(this.net{i}.Ws:this.net{i}.We),this.net{i}.Wshape)*this.O{inInd}...
                            + this.theta(this.net{i}.bs:this.net{i}.be)*this.net{i}.ones;
                    case 'relu'
                        ind = (real(this.O{inInd}) >= 0 & imag(this.O{inInd}) >= 0);
                        this.O{outInd} = this.O{inInd}.*ind;
                    case 'relu_or'
                        ind = (real(this.O{inInd}) >= 0  | imag(this.O{inInd}) >= 0);
                        this.O{outInd} = this.O{inInd}.*ind;
                    case 'exp'
                        this.O{outInd} = exp(this.O{inInd});
                    case 'abs_relu'
                        ind = (abs(this.O{inInd}) >= this.net{i}.threshold);
                        this.O{outInd} = this.O{inInd}.*ind;
                    case 'tanh'
                        this.O{outInd} = tanh(this.O{inInd});
                    case 'abs'
                         this.O{outInd} = this.O{inInd};
                        this.O{outInd}(this.net{i}.channels{:}) = abs(this.O{inInd}(this.net{i}.channels{:}));
                    case 'abs_from_real'
                        s = size(this.O{inInd});
                        for l=2:2:s(end-1)
                            this.O{outInd}(:,:,l/2,:)=(this.O{inInd}(:,:,l-1,:).^2+this.O{inInd}(:,:,l,:).^2).^(1/2);
                        end
                     case 'sq_abs'
                        this.O{outInd} = abs(this.O{inInd}).^2;
                    case 'clamp'
                        this.O{outInd} = min(1,max(0,this.O{inInd}));
                    case 'loss'
                        this.O{outInd} = this.net{i}.loss.LossAndErr(this.O{inInd(1)},this.O{inInd(2)});
                    case 'concat'
                        this.O{outInd} = cat(this.net{i}.dim,this.O{inInd(1)},this.O{inInd(2)});
                    case 'pad'
                        blobSize = size(this.O{inInd});
                        this.O{outInd}(1:blobSize(1),1:blobSize(2),:,:) = this.O{inInd};
                    case 'elementwiseProd'
                        this.O{outInd} = this.O{inInd(1)} .* this.O{inInd(2)};
                    case 'add'
                        this.O{outInd} = this.net{i}.alpha * this.O{inInd(1)} + this.net{i}.beta *this.O{inInd(2)};
                    case 'real2comp'
                        this.O{outInd} = this.O{inInd}(:,:,1:2:end,:)+1i*this.O{inInd}(:,:,2:2:end,:);
                    otherwise
                        assert(false,'Unknown Layer type')
                end
            end
        end
        
        % backward function
        function backward(this,lam)
            
            this.dtheta = this.theta-this.theta;
            
            for i=this.nLayers:-1:1
                outInd = this.net{i}.outInd;
                inInd = this.net{i}.inInd;
                
                if isequal(this.net{i}.type,'affine')
                    this.dtheta(this.net{i}.Ws:this.net{i}.We) = reshape( this.delta{outInd} * this.O{inInd}' , this.net{i}.We-this.net{i}.Ws+1 , 1); % Nabla_W = delta*O{i-1}';
                    this.dtheta(this.net{i}.bs:this.net{i}.be) = reshape( sum(this.delta{outInd},2).' , this.net{i}.be-this.net{i}.bs+1 , 1); % Nabla_b = sum(delta');
                end
                
                if ~(this.net{i}.needBackward)
                    continue;
                end
                
                switch this.net{i}.type
                    case 'loss'
                        this.delta{inInd(1)} = this.net{i}.loss.Grad(this.O{inInd(1)},this.O{inInd(2)});
                    case 'duplicate'
                        this.delta{inInd} = this.delta{outInd(1)};
                        for j=2:length(outInd)
                            this.delta{inInd} = this.delta{inInd} + this.delta{outInd(j)};
                        end
                    case 'im2col'
                        % method I
                        tmp = cumsum(this.delta{outInd}(this.net{i}.sortedInd));
                        this.delta{inInd} = reshape([tmp(1) ; diff(tmp(this.net{i}.I))] , size(this.O{inInd}));
                    case 'max'
                        s = [size(this.O{inInd},1) 1];
                        tmp = (repmat(this.O{outInd},s) == this.O{inInd});
                        this.delta{inInd} = repmat(this.delta{outInd} ./ sum(tmp),s) .* tmp;
                    case 'mean'
                        s = [size(this.O{inInd},1) 1];
                        this.delta{inInd} = repmat(this.delta{outInd} / s(1),s);
                        
                    case 'reshape'
                        this.delta{inInd} = reshape(this.delta{outInd}, this.net{i}.oldshape);
                    case 'permute'
                        this.delta{inInd} = permute(this.delta{outInd}, this.net{i}.oldshape);
                    case 'affine'
                        this.delta{inInd} = reshape(this.theta(this.net{i}.Ws:this.net{i}.We),this.net{i}.Wshape).' * this.delta{outInd};
                    case 'relu'
                        ind = (real(this.O{inInd}) >= 0 & imag(this.O{inInd}) >= 0);
                        this.delta{inInd} = this.delta{outInd} .* ind;
                    case 'relu_or'
                        ind = (real(this.O{inInd}) >= 0 | imag(this.O{inInd}) >= 0);
                        this.delta{inInd} = this.delta{outInd} .* ind;
                    case 'exp'
                        this.delta{inInd} = this.delta{outInd} .* this.O{outInd};
                    case 'abs_relu'
                        ind = (abs(this.O{inInd}) >= this.net{i}.threshold);
                        this.delta{inInd} = this.delta{outInd} .* ind;
                    case 'tanh'
                        this.delta{inInd} = this.delta{outInd} .* conj(sech(this.O{inInd}).^2);%d/dx (tanh(x)) = sech(x)^2
                    case 'abs'
                        this.delta{inInd} = this.delta{outInd};
                        this.delta{inInd}(this.net{i}.channels{:}) = ...
                            this.delta{outInd}(this.net{i}.channels{:}) .* this.O{inInd}(this.net{i}.channels{:}) ./ this.O{outInd}(this.net{i}.channels{:});%d/dx (abs(x)) = x/abs(x)
                    case 'abs_from_real'
                        s = size(this.O{inInd});
                        for l=2:2:s(end-1)
                            this.delta{inInd}(:,:,l-1:l,:) = ...
                                bsxfun(@mtimes,this.delta{outInd}(:,:,l/2,:)./this.O{outInd}(:,:,l/2,:), this.O{inInd}(:,:,l-1:l,:));
                        end
                    case 'sq_abs'
                        this.delta{inInd} = this.delta{outInd} .* (2 * this.O{inInd}) ;%d/dx (abs(x)^2) = 2*x
                    case 'clamp'
                        this.delta{inInd} = this.delta{outInd} .* ((this.O{outInd} > 0) & (this.O{outInd} < 0));
                    case 'concat'
                        tmp = cell(length(size(this.O{outInd})),1); 
                        for j=1:length(tmp), tmp{j} = ':'; end;
                        tmp{this.net{i}.dim} = 1:size(this.O{inInd(1)},this.net{i}.dim);
                        this.delta{inInd(1)} = this.delta{outInd}(tmp{:});
                        tmp{this.net{i}.dim} = (1+size(this.O{inInd(1)},this.net{i}.dim)):size(this.O{outInd},this.net{i}.dim);
                        this.delta{inInd(2)} = this.delta{outInd}(tmp{:});
                    case 'pad'
                        blobSize = size(this.O{inInd});
                        this.delta{inInd} = this.delta{outInd}(1:blobSize(1),1:blobSize(2),:,:);
                    case 'elementwiseProd'
                        this.delta{inInd(1)} = this.delta{outInd} .* this.O{inInd(2)};
                        this.delta{inInd(2)} = this.delta{outInd} .* this.O{inInd(1)};
                    case 'add'
                        this.delta{inInd(1)} = this.net{i}.alpha * this.delta{outInd};
                        this.delta{inInd(2)} = this.net{i}.beta  * this.delta{outInd};
                    case 'real2comp'
                        this.delta{inInd} = this.O{inInd} - this.O{inInd};
                        for l=1:size(this.delta{outInd},3)
                            this.delta{inInd}(:,:,2*l-1,:) = real(this.delta{outInd}(:,:,l,:));
                            this.delta{inInd}(:,:,2*l,:) = imag(this.delta{outInd}(:,:,l,:));
                        end
                    otherwise
                        assert(false,'Unknown Layer type')
                end
                
            end
            
            % and add the regularization gradient
            this.dtheta = this.dtheta + lam*this.theta;
            
        end
        
        function [init_theta,AllLoss,seed] = OptimizationWrapper(this,initFunc,optFunc,T_trial,threshold,limit,T)
            stop_condition.T = T_trial;
            stop_condition.threshold = threshold;
            for i=1:limit
                initFunc();
                init_theta = this.theta;
                [losses,stoped,seed] = optFunc(T,stop_condition);
                if ~stoped
                    AllLoss = losses;
                    return ;
                end
            end
            AllLoss = NaN;
        end
                
            
        % SGD with Nesterov's momentum
        function [AllLoss,stoped,seed] = Nesterov(this,T,learningRate,mu,lam,printIter,projection,adaptive,losses_hist,init_lr,stop_condition,seed)
            
            prev_losses = zeros(2,losses_hist);
            stoped = 0;
            prev_loss = Inf;
            thresh = 1e-2;
            m = this.net{1}.data.m;
            
            Loss=this.O{this.lossInd}-this.O{this.lossInd};
            AllLoss=repmat(Loss,1,floor(T/printIter));
            AllLoss = [AllLoss ; AllLoss ; AllLoss];
            AllLoss(:) = NaN;
            history = this.theta-this.theta;
            eta = init_lr;
            if nargin<12
                seed = randi(10000,1,1);
            end
            rng(seed);
            
            for t=1:T

                % momentum
                history = mu*history;
                this.theta = this.theta + history;
                
                % choose mini batch
                i = randi(m,1,1);
                % forward backward
                this.forward(i);
                this.backward(lam);
                % update                  
                if adaptive
                    if ((rem(t,printIter)==0) && (t/printIter >= losses_hist))  || isempty(eta)
                        fprintf('Iteration: %d, checking the lr\n',t);
%                         [eta,prev_loss,thresh] = learningRate(eta,this.theta,prev_loss,thresh);
                        eta = learningRate(eta,prev_losses);
                                           
                    end
                else
                        eta = learningRate(t);
                end
%                 fprintf('t: %d, mean: %d, max: %d, min: %d, var: %d\n',t,mean(eta*this.dtheta),max(eta*this.dtheta),min(eta*this.dtheta),var(eta*this.dtheta));
                x = this.theta - eta*this.dtheta;
                if sum(isnan(x))>0 || sum(isinf(x))>0
                    fprintf('Breaking at iteration # %d\n',t);
                    break
                end
                
                this.theta = x;
                history = history - eta*this.dtheta;
                %projection
                switch projection.type
                    case 'norm'
                        for i=1:length(this.net)
                            if strcmp(this.net{i}.type,'affine')
                                W = this.theta(this.net{i}.Ws:this.net{i}.We); 
                                b = this.theta(this.net{i}.bs:this.net{i}.be); 
                                if norm(W)>projection.bound
        %                             fprintf('Normalizing W: iteration: %d, layer: %d, norm (before): %d\n'...
        %                                 ,t,i,norm(W));
                                    W = projection.bound*0.1*W./norm(W);
                                    this.theta(this.net{i}.Ws:this.net{i}.We) = W;
                                end
                                if norm(b)>projection.bound
        %                             fprintf('Normalizing b: iteration: %d, layer: %d, norm (before): %d\n'...
        %                                 ,t,i,norm(b));
                                    b = projection.bound*0.1*b./norm(b);
                                    this.theta(this.net{i}.bs:this.net{i}.be) = b; 
                                end
                            end
                        end
                    case 'phase'
                        for i=1:length(this.net)
                            if strcmp(this.net{i}.type,'affine')
                                W = (reshape(this.theta(this.net{i}.Ws:this.net{i}.We),this.net{i}.Wshape)); 
                                b = (this.theta(this.net{i}.bs:this.net{i}.be));
                                ang = angle([W b]);
                                avg_phase = sum(ang,2)./size(ang,2);
                                avg_phase = exp(1i*avg_phase);
                                new_W = bsxfun(@rdivide,W, avg_phase);
                                new_b = bsxfun(@rdivide,b, avg_phase);
                                this.theta(this.net{i}.Ws:this.net{i}.We) = new_W(:);
                                this.theta(this.net{i}.bs:this.net{i}.be) = new_b(:);
                            end
                        end
                end
                
                   
                % calculate statistics for printing
%                 Loss = 0.9*Loss + 0.1*this.O{this.lossInd};
                Loss = this.O{this.lossInd};
%                 real_loss = Loss;
                if (rem(t,printIter)==0)
                    
                    real_loss = this.calcLossAndErr();
                    if ~isempty(this.testNet)
                        this.testNet.setTheta(this.theta);
                        test_loss = this.testNet.calcLossAndErr();
                    else
                        test_loss = [0 ; 0];
                    end
                    AllLoss(:,t/printIter) = [Loss;real_loss;test_loss];
                    if adaptive
                        prev_losses = [prev_losses,gather(Loss)];
                        prev_losses = prev_losses(:,2:end); 
                    end
                    fprintf('Current Error   - Nesterov Iter: %d: ',t); for i=1:length(Loss), fprintf('%f ',Loss(i)); end; fprintf('\n');
                    fprintf('Training Error  - Nesterov Iter: %d: ',t); for i=1:length(Loss), fprintf('%f ',real_loss(i)); end; fprintf('\n');
                    fprintf('Test Error      - Nesterov Iter: %d: ',t); for i=1:length(Loss), fprintf('%f ',test_loss(i)); end; fprintf('\n');


%                 elseif  t ==1
%                     tic;real_loss = this.calcLossAndErr();toc
% 
%                     fprintf('Current Error -   Nesterov Iter: %d: ',t); for i=1:length(Loss), fprintf('%f ',Loss(i)); end; fprintf('\n');
%                     fprintf('Training Error  - Nesterov Iter: %d: ',t); for i=1:length(Loss), fprintf('%f ',real_loss(i)); end; fprintf('\n');
% 
                end
                if t==stop_condition.T
                    if Loss(end)>= stop_condition.threshold
                        stoped = 1;
                        return;
                    end
                end
                
            end
        end

        % vanilla SGD solver
        function [AllLoss] = SGD(this,T,learningRate,lam,printIter)
            
            m = this.net{1}.data.m;
            
            Loss=this.O{this.lossInd}-this.O{this.lossInd};
            AllLoss=repmat(Loss,1,floor(T/printIter));
                        
            for t=1:T
                % choose mini batch
                i = randi(m,1,1);
                % forward backward
                this.forward(i);
                this.backward(lam);
                % update
                eta = learningRate(t);
                this.theta = this.theta - eta*this.dtheta;
                
                % calculate statistics for printing
                Loss = 0.9*Loss + 0.1*this.O{this.lossInd};
                if (rem(t,printIter)==0)
                    AllLoss(:,t/printIter) = Loss;
                    fprintf(1,'SGD Iter: %d: ',t); for i=1:length(Loss), fprintf(1,'%f ',Loss(i)); end; fprintf(1,'\n');
                end
                
                
            end
        end

        % SDCA solver 
        function [AllLoss] = SDCA(this,alpha,T,eta,lam,printIter)
            
            % help variables
            m = this.net{1}.data.m;
            if ~isempty(alpha)
                % initialize primal from dual
                [d,n] = size(alpha);
                assert(n == m);
                assert(d == length(this.theta));
            else
                % initialize by random dual variabls
                d = length(this.theta);
                n = m;
                alpha = randn(d,n,'single')*lam;
            end
            this.theta = this.mygpuArray(single(mean(alpha,2)/lam));
            beta = eta*lam*n;

            Loss=this.O{this.lossInd}-this.O{this.lossInd};
            AllLoss=repmat(Loss,1,floor(T/printIter));
                        
            for t=1:T
                % choose mini batch
                i = randi(m,1,1);
                galpha = this.mygpuArray(alpha(:,i));
                % forward backward
                this.forward(i);
                this.backward(0);
                
                % update
                v = this.dtheta+galpha;
                galpha = galpha - beta*v;
                this.theta = this.theta - eta*v;
                alpha(:,i) = this.mygather(galpha);
                
                % calculate statistics for printing
                Loss = 0.9*Loss + 0.1*this.O{this.lossInd};
                if (rem(t,printIter)==0)
                    AllLoss(:,t/printIter) = Loss;
                    fprintf(1,'SDCA Iter: %d: ',t); for i=1:length(Loss), fprintf(1,'%f ',Loss(i)); end; fprintf(1,'\n');
                end
                
                
            end
        end
        
        function [Loss,pred] = calcLossAndErr(this)
            m = this.net{1}.data.m;
            pred = [];
            Loss = this.O{this.lossInd}-this.O{this.lossInd};
            for i=1:m
                this.forward(i);
                Loss = Loss + this.O{this.lossInd};
                pred = [pred this.O{this.lossInd-1}];
            end
            Loss = Loss/m;
        end
        
        
        function setTheta(this,newtheta)
            this.theta = this.mygpuArray(newtheta);
        end
        
        function initializeWeights(this,varargin)
        % Initialize the weights for every affine layer.  
            imag_coeff = 1;
            tot_coeff = 1/2;
            bound_coeff = Inf;
            for i=1:nargin/2
                switch varargin{2*i-1}
                    case 'imag_coeff'
                        imag_coeff = varargin{2*i};
                    case 'tot_coeff'
                        tot_coeff = varargin{2*i};
                     case 'bound_coeff'
                        bound_coeff = varargin{2*i};
                end
            end
                
            for i=1:length(this.net)
                if strcmp(this.net{i}.type,'affine')                     
                        blobDim = size(this.O{this.net{i}.inInd});
                        nrows = blobDim(1);
                        bound = min(bound_coeff,sqrt(3)/sqrt(nrows)); % 0.1 so that |W|=Sqrt(WRe/2 + i WIm/2)<0.1 according to "Fully". Using with tot_coeff=1/2 and imag_coeff=1;
                        fprintf('Xaviar: %g, Bound: %g\n',sqrt(3)/sqrt(nrows),bound);

                        if this.net{i}.comp
                            WRe = (rand(this.net{i}.Wshape)-0.5)*bound;
                            WIm = (rand(this.net{i}.Wshape)-0.5)*bound;
                            W = tot_coeff*(WRe+1i*imag_coeff*WIm);
                        else
                            W = (rand(this.net{i}.Wshape)-0.5)* bound;
                        end
                        this.theta(this.net{i}.Ws:this.net{i}.We)= this.mygpuArray(W(:));
                        b = zeros(size(this.net{i}.Wshape,1),1) + this.net{i}.bias_filler;
                        this.theta(this.net{i}.bs:this.net{i}.be) = this.mygpuArray(b);
                end
            end
        end
        
    
    
    function initializeWeightsGaussian(this,sigma)
        % Initialize the weights for every affine layer.                
            for i=1:length(this.net)
                if strcmp(this.net{i}.type,'affine')                     
                        if this.net{i}.comp
                            W =  mvnrnd([0 0], sigma*eye(2), this.net{i}.We-this.net{i}.Ws+1);
                            W = W(:,1)+1i*W(:,2);
                        else
                            W =  mvnrnd(0, sigma, this.net{i}.We-this.net{i}.Ws+1);
                        end
                        this.theta(this.net{i}.Ws:this.net{i}.We)= this.mygpuArray(W(:));
                        b = zeros(size(this.net{i}.Wshape,1),1) + this.net{i}.bias_filler;
                        this.theta(this.net{i}.bs:this.net{i}.be) = this.mygpuArray(b);
                end
            end
        end
        
    end
    
    % Private methods for initializing the network
    methods (Access = private)
        function initializeNet(this,netD,atGPU)
            % construct a network (net,theta) and initialize the network based on a
            % description given in netD
            
            
            % find maximal value of Oind and required number of layers
            maxOind = 0;
            for i=1:length(netD)
                maxOind = max(maxOind,max(netD{i}.outInd));
            end
            lenO = maxOind;
            this.nLayers = length(netD);
            for i=1:length(netD)
                if strcmp(netD{i}.type,'conv')
                    this.nLayers = this.nLayers + 3;
                    lenO = lenO + 3;
                elseif strcmp(netD{i}.type,'maxpool')
                    this.nLayers = this.nLayers + 2;
                    lenO = lenO + 2;
                elseif strcmp(netD{i}.type,'avgpool')
                    this.nLayers = this.nLayers + 2;
                    lenO = lenO + 2;
                end
            end
            
            
            % initialize
            this.net = cell(this.nLayers,1);
            this.O = cell(lenO,1);
            layerInd = 0;
            this.theta = [];
            needBack = false(lenO,1);
            
            for i=1:length(netD)
                
                Oind = netD{i}.outInd;
                
                % determine the needBackward flag
                if ~strcmp(netD{i}.type,'input')
                    inInd = netD{i}.inInd;
                    needBackward = sum(needBack(inInd))>0;
                end
                
                switch netD{i}.type
                    case 'input'
                        this.O{Oind} = zeros(netD{i}.blobSize,'single');
                        layerInd = layerInd+1;
                        if ~isfield(netD{i},'maxExamples')
                            maxExamples  = Inf;
                        else
                            maxExamples  = netD{i}.maxExamples;
                        end
                        this.net{layerInd} = struct('type','input','outInd',Oind,'inInd',0,'needBackward',false,'data',...
                            dataClass(netD{i}.fName,netD{i}.dataType,netD{i}.blobSize,netD{i}.scale,maxExamples,atGPU,...
                            netD{i}.dataFormat,netD{i}.comp,netD{i}.shift));
                        
                    case 'duplicate'
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','duplicate','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        for j=Oind
                            this.O{j} = this.O{inInd};
                        end
                        needBack(Oind) = needBack(inInd);
                        
                    case 'concat'
                        layerInd = layerInd+1;
                        this.net{layerInd} = netD{i}; this.net{layerInd}.needBackward = needBackward;
                        this.O{Oind} = cat(netD{i}.dim,this.O{inInd(1)},this.O{inInd(2)});
                        needBack(Oind) = sum(needBack(inInd))>0;
                        
                    case 'conv'
                        if ~isfield(netD{i},'comp')
                            netD{i}.comp = 0;
                        end
                        originalBlobDimSize = size(this.O{inInd});
                        
                        % construct im2col layer
                        maxOind = maxOind + 1;
                        [layer,blobDim,height,width] = this.constructIm2ColLayer(netD{i}.kernelsize,netD{i}.stride,originalBlobDimSize,needBackward,maxOind,inInd,true);
                        layerInd = layerInd+1;
                        this.net{layerInd} = layer;
                        this.O{maxOind} = zeros(blobDim,'single');
                        needBack(maxOind) = needBack(inInd);
                        
                        % then affine layer
                        nOut = netD{i}.nOutChannels;
                        W = (rand(netD{i}.nOutChannels,blobDim(1))-0.5)/ sqrt(blobDim(1)) * sqrt(3);
                        Wind = length(this.theta)+(1:length(W(:)));
                        this.theta = [this.theta ; W(:)];
                        b = zeros(nOut,1) + netD{i}.bias_filler;
                        bind = length(this.theta)+(1:length(b));
                        this.theta = [this.theta ; b];
                        layerInd = layerInd+1;
                        maxOind = maxOind + 1;
                        this.net{layerInd} = struct('type','affine','outInd',...
                            maxOind,'inInd',maxOind-1,'ones',this.mygpuArray(ones(1,blobDim(2),'single')),...
                            'Ws',min(Wind),'We',max(Wind),'Wshape',size(W),'bs',min(bind),'be',max(bind),...
                            'needBackward',needBackward,'comp',netD{i}.comp,'bias_filler',netD{i}.bias_filler);
                        this.O{maxOind} = zeros(size(W,1),blobDim(2),'single');
                        blobDim = [size(W,1) blobDim(2)];
                        needBack(maxOind) = true;
                        
                        
                        % and then reshape and permute layers
                        % currently, the order in memory is
                        %   (channels,height,width,items)
                        % we want it to be
                        %   (height,width,channels,items)
                        channels = netD{i}.nOutChannels;
                        items = originalBlobDimSize(4);
                        layerInd = layerInd+1;
                        maxOind = maxOind + 1;
                        this.net{layerInd} = struct('type','reshape','outInd',maxOind,'inInd',maxOind-1,'newshape',[channels height width items],'oldshape',[channels height*width*items],'needBackward',true);
                        this.O{maxOind} = reshape(this.O{maxOind-1},[channels height width items]);
                        needBack(maxOind) = true;
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','permute','outInd',Oind,'inInd',maxOind,'newshape',[2 3 1 4],'oldshape',[3 1 2 4],'needBackward',true);
                        this.O{Oind} = permute(this.O{maxOind},[2 3 1 4]);
                        needBack(Oind) = true;
                        
                        
                    case 'flatten'
                        
                        blobDim = size(this.O{inInd});
                        newshape = [prod(blobDim(1:3)) blobDim(4)];
                        layerInd = layerInd+1;
                        this.O{Oind} = zeros(newshape,'single');
                        this.net{layerInd} = struct('type','reshape','outInd',Oind,'inInd',inInd,'newshape',newshape,'oldshape',blobDim,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'affine'
                        if ~isfield(netD{i},'comp')
                            netD{i}.comp = 0;
                        end
                        
                        blobDim = size(this.O{inInd});
                        ncol = blobDim(2);
                        nrows = blobDim(1);
                        W = (rand(netD{i}.nOutChannels,nrows)-0.5)/ sqrt(nrows) * sqrt(3);
                        if netD{i}.comp
                            WIm = (rand(netD{i}.nOutChannels,nrows)-0.5)/ sqrt(nrows) * sqrt(3);
                            W = (W+1i*WIm)/2;
                        end

                        Wind = length(this.theta)+(1:length(W(:)));
                        this.theta = [this.theta ; W(:)];
                        b = zeros(netD{i}.nOutChannels,1) + netD{i}.bias_filler;
                        bind = length(this.theta)+(1:length(b));
                        this.theta = [this.theta ; b];
                        this.O{Oind} = zeros(size(W,1),ncol,'single');
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','affine','outInd',Oind,...
                            'inInd',inInd,'ones',this.mygpuArray(ones(1,ncol,'single')),...
                            'Ws',min(Wind),'We',max(Wind),'Wshape',size(W),'bs',min(bind),...
                            'be',max(bind),'needBackward',needBackward,'comp',netD{i}.comp,'bias_filler',netD{i}.bias_filler);
                        
                        needBack(Oind) = true;
                        
                    case 'maxpool'
                        
                        originalBlobDimSize = size(this.O{inInd});
                        
                        % construct im2col layer
                        maxOind = maxOind + 1;
                        [layer,blobDim,height,width] = this.constructIm2ColLayer(netD{i}.kernelsize,netD{i}.stride,originalBlobDimSize,needBackward,maxOind,inInd,false);
                        this.O{maxOind} = zeros(blobDim,'single');
                        layerInd = layerInd+1;
                        this.net{layerInd} = layer;
                        needBack(maxOind) = needBack(inInd);
                        
                        
                        % then max layer
                        maxOind = maxOind + 1;
                        blobDim = [1 blobDim(2)];
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','max','outInd',maxOind,'inInd',maxOind-1,'needBackward',needBackward);
                        this.O{maxOind} = zeros(blobDim,'single');
                        needBack(maxOind) = needBack(inInd);
                        
                        % and then reshape
                        channels = originalBlobDimSize(3);
                        items = originalBlobDimSize(4);
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','reshape','outInd',Oind,'inInd',maxOind,'newshape',[height width channels items],'oldshape',blobDim,'needBackward',needBackward);
                        this.O{Oind} = reshape(this.O{maxOind},[height width channels items]);
                        needBack(Oind) = needBack(inInd);
                        
                        
                    case 'avgpool'
                        
                        originalBlobDimSize = size(this.O{inInd});
                        
                        % construct im2col layer
                        maxOind = maxOind + 1;
                        [layer,blobDim,height,width] = this.constructIm2ColLayer(netD{i}.kernelsize,netD{i}.stride,originalBlobDimSize,needBackward,maxOind,inInd,false);
                        this.O{maxOind} = zeros(blobDim,'single');
                        layerInd = layerInd+1;
                        this.net{layerInd} = layer;
                        needBack(maxOind) = needBack(inInd);
                        
                        % then mean layer
                        maxOind = maxOind + 1;
                        blobDim = [1 blobDim(2)];
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','mean','outInd',maxOind,'inInd',maxOind-1,'needBackward',needBackward);
                        this.O{maxOind} = zeros(blobDim,'single');
                        needBack(maxOind) = needBack(inInd);
                        
                        % and then reshape
                        channels = originalBlobDimSize(3);
                        items = originalBlobDimSize(4);
                        layerInd = layerInd+1;
                        this.net{layerInd} = struct('type','reshape','outInd',Oind,'inInd',maxOind,'newshape',[height width channels items],'oldshape',blobDim,'needBackward',needBackward);
                        this.O{Oind} = reshape(this.O{maxOind},[height width channels items]);
                        needBack(Oind) = needBack(inInd);
                        
                        
                    case 'relu'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd};
                        this.net{layerInd} = struct('type','relu','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                   
                    case 'relu_or'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd};
                        this.net{layerInd} = struct('type','relu_or','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                    
                    case 'exp'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd};
                        this.net{layerInd} = struct('type','exp','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                        
                    case 'abs_relu'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd};
                        this.net{layerInd} = struct('type','abs_relu','outInd',Oind,...
                            'inInd',inInd,'needBackward',needBackward,'threshold',netD{i}.threshold);
                        needBack(Oind) = needBack(inInd);
                   
                    case 'tanh'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd};
                        this.net{layerInd} = struct('type','tanh','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                    case 'abs'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd};
                        this.net{layerInd} = struct('type','abs','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        if isfield(netD{i},'channels')
                            this.net{layerInd}.channels = netD{i}.channels;
                        else
                            this.net{layerInd}.channels = {':'};
                        end
                    case 'abs_from_real'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd}(:,:,1:2:end,:);
                        this.net{layerInd} = struct('type','abs_from_real','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                    case 'sq_abs'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd};
                        this.net{layerInd} = struct('type','sq_abs','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                    case 'clamp'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd};
                        this.net{layerInd} = struct('type','clamp','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'reshape'
                        layerInd = layerInd+1;
                        this.O{Oind} = reshape(this.O{inInd},netD{i}.newshape);
                        this.net{layerInd} = struct('type','reshape','outInd',Oind,'inInd',inInd,'newshape',netD{i}.newshape,'oldshape',size(this.O{inInd}),'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'permute'
                        [~,ind] = sort(netD{i}.newshape);
                        layerInd = layerInd+1;
                        this.O{Oind} = permute(this.O{inInd},netD{i}.newshape);
                        this.net{layerInd} = struct('type','permute','outInd',Oind,'inInd',inInd,'newshape',netD{i}.newshape,'oldshape',ind,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'pad'
                        layerInd = layerInd+1;
                        blobSize = size(this.O{inInd});
                        this.O{Oind} = zeros(blobSize(1)+netD{i}.amount,blobSize(2)+netD{i}.amount,blobSize(3),blobSize(4),'single');
                        this.net{layerInd} = struct('type','pad','outInd',Oind,'inInd',inInd,'amount',netD{i}.amount,'needBackward',needBackward);
                        needBack(Oind) = needBack(inInd);
                        
                    case 'elementwiseProd'                        
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd(1)};
                        this.net{layerInd} = struct('type','elementwiseProd','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = sum(needBack(inInd))>0;
                    
                    case 'add'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd(1)};
                        this.net{layerInd} = struct('type','add','outInd',Oind,'inInd',inInd,'alpha',netD{i}.alpha,'beta',netD{i}.beta,'needBackward',needBackward);
                        needBack(Oind) = sum(needBack(inInd))>0;
                    
                    case 'real2comp'
                        layerInd = layerInd+1;
                        this.O{Oind} = this.O{inInd}(:,:,1:2:end,:);
                        this.net{layerInd} = struct('type','real2comp','outInd',Oind,'inInd',inInd,'needBackward',needBackward);
                        needBack(Oind) = sum(needBack(inInd))>0;

                    case 'loss'
                        layerInd = layerInd+1;
                        if ~isfield(netD{i},'wrapper')
                            wrapper = false;
                        else
                            wrapper = netD{i}.wrapper;
                        end
                        if wrapper
                            lc =  lossClassWrapper(netD{i}.lossType,netD{i}.scales);
                        else
                            lc = lossClass(netD{i}.lossType);
                        end
                        this.net{layerInd} = struct('type','loss','outInd',Oind,'inInd',inInd,'loss',lc,'needBackward',needBackward);
                        this.O{Oind} = this.net{layerInd}.loss.LossAndErr(this.O{inInd(1)},this.O{inInd(2)});
                        needBack(Oind) = needBack(inInd(1));
                        this.lossInd = Oind;
                        
                    otherwise
                        assert(false,'Unknown Layer type')
                end
            end
            
            
            % make everything single, atGPU, and allocate delta
            this.theta = this.mygpuArray(single(this.theta));
            for i=1:length(this.O)
                this.O{i} = this.mygpuArray(this.O{i});
            end
            
        end
        
        function [layer,blobDim,outHeight,outWidth] = constructIm2ColLayer(this,ksize,kstride,blobDim,needBackward,Oind,inInd,isConv)
            
            B = reshape((1:(prod(blobDim))),blobDim);
            C = [];
            
            if isConv
                for t=1:blobDim(4)
                    w=0;
                    while (w+ksize <= blobDim(2))
                        h=0;
                        while (h+ksize <= blobDim(1))
                            C = [C reshape(B(h+(1:ksize),w+(1:ksize),:,t),ksize*ksize*blobDim(3),1)];
                            h = h+kstride;
                        end
                        w=w+kstride;
                    end
                end
                outHeight =  h/kstride;
                outWidth = w/kstride;
            else % for pooling layer
                outHeight = ceil((blobDim(1) - ksize)/kstride) + 1;
                outWidth = ceil((blobDim(2) - ksize)/kstride) + 1;
                for t=1:blobDim(4)
                    for c=1:blobDim(3)
                        for ww=1:outWidth
                            ws = (ww-1)*kstride + 1;
                            we = min(size(B,2),ws-1+ksize);
                            Iw = zeros(ksize,1) + we;
                            Iw((ws:we)-ws+1) = (ws:we);
                            for hh=1:outHeight
                                hs = (hh-1)*kstride + 1;
                                he = min(size(B,1),hs-1+ksize);
                                Ih = zeros(ksize,1) + he;
                                Ih((hs:he)-hs+1) = (hs:he);
                                C = [C reshape(B(Ih,Iw,c,t),ksize*ksize,1)];
                            end
                        end
                    end
                end
            end
            [val,ind] = sort(C(:));
            I = [find(val(1:end-1) ~= val(2:end)) ; length(val)];
            
            % method II -- not implmenented properly
            % backwardMat = zeros(ksize*ksize,prod(blobDim));
            % backwardMat(1,1) = 1;
            % for j=2:length(I)
            %     backwardMat(1:length(ind((I(j-1)+1):I(j))),j) = ind((I(j-1)+1):I(j));
            % end
            % J = find(backwardMat>0); bI = backwardMat(J);
            %
            % For the above method we will need:
            %this.net{i}.backwardMat(this.net{i}.J) = delta(this.net{i}.bI);
            %delta = reshape( sum(this.net{i}.backwardMat) , size(this.O{this.net{i}.inInd}));
            
            blobDim = size(C);
            layer = struct('type','im2col','outInd',Oind,'inInd',inInd,'im2colInd',this.mygpuArray(uint32(C)),'sortedInd',this.mygpuArray(uint32(ind)),'I',this.mygpuArray(uint32(I)),'needBackward',needBackward);
            %layer = struct('type','im2col','im2colInd',mygpuArray(uint32(C)),'backwardMat',mygpuArray(zeros(size(backwardMat),'single')),'J',mygpuArray(uint32(J)),'bI',mygpuArray(uint32(bI)),'sortedInd',mygpuArray(uint32(ind)),'I',mygpuArray(uint32(I)),'inInd',int32(inInd),'outInd',int32(outInd),'needBackward',needBackward);
            
        end
    end
    
    
end
