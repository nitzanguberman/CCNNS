% a simple demonstration of ConvNet code

%%

% describe dataLocati
clear all; clc
dataLocation = 'F:\simcep\simcep\simple\exp1';

atGPU = false;
[~,compName] = unix('hostname');
if isequal(compName(1:end-1),'deep-01')
    atGPU = true; 
end

%%

% initialize a network
lenet = { ...
    struct('type','input','outInd',1,'inInd',0,'blobSize',[15 15 1 100],...
    'fName',fullfile(dataLocation ,'cells_s_150_N_50_p_0_2_grads_fixed.bin'),...
    'scale',1/32,'dataType','uint8','maxExamples',100,'dataFormat','bin','comp',1,'shift',-(4+4i)), ...
    struct('type','input','outInd',2,'inInd',0,'blobSize',[2 100],...
    'fName',fullfile(dataLocation ,'gt_boolean_exp1_thresh_10.bin'),'scale',1,...
    'dataType','uint8','dataFormat','bin','comp',0,'shift',0), ...
    struct('type','conv','outInd',3,'inInd',1,'kernelsize',5,'stride',1,....
    'nOutChannels',5,'bias_filler',0,'comp',1),...
    struct('type','relu','outInd',5,'inInd',3),...
    struct('type','maxpool','outInd',6,'inInd',5,'kernelsize',2,'stride',1),...
    struct('type','conv','outInd',7,'inInd',6,'kernelsize',5,'stride',1,....
    'nOutChannels',2,'bias_filler',0,'comp',1),...
    struct('type','relu','outInd',8,'inInd',7),...
    struct('type','maxpool','outInd',9,'inInd',8,'kernelsize',6,'stride',6),...
    struct('type','flatten','outInd',10,'inInd',9),...
    struct('type','sq_abs','outInd',11,'inInd',10),...
    struct('type','loss','outInd',12,'inInd',[11 2],'lossType','MCLogLoss') };

tic;
cnn = ConvNet(lenet,atGPU);
toc;
%%
testlenet = lenet;
testlenet{1}.fName = fullfile(dataLocation ,'cells_s_150_N_50_p_0_2_grads_fixed_test.bin');
testlenet{2}.fName = fullfile(dataLocation, 'gt_boolean_exp1_thresh_10_test.bin');
tic;
testNet = ConvNet(testlenet,atGPU);
toc;
cnn.setTestNet(testNet);

%% 
% show some images
x = testNet.net{1}.data.get(1); y = testNet.net{2}.data.get(1); [~,bla] = max(y);
figure;
for i=1:10
    subplot(3,4,i);
    quiverplot(squeeze(x(:,:,:,i)));
    title(sprintf('%d',bla(i)-1)); 
end


%%

% Train using SGD with Nesterov's momentum
T = 10000; printIter = 100; %lam =  single(0.000);
projection.type = NaN;
T_trial = 700;
threshold = 0.4;
limit = 100;
lam = 0;
results = {};
%%
% for i=1:40
%     for lam=[0,10e-7,10e-5,10e-3,10e-1,1,10]
% for sigma = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1]
        for base_lr = 0.01%[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,0.5,1,5]%fliplr([1e-2,1e-1,5e-1,1,5,10,20])
            for mu=0.9%[0,0.5,0.9]
                for t=1:40

%             for lr_coeff = [0.0001,0.001,0.01,0.1,0.5,1]
               fprintf('Training trial number: %d/40\n',t);
                res.mu = mu;
                res.base_lr = base_lr;
                
         
%                 res.lr_coeff = lr_coeff;
%                 res.sigma = sigma;
    %             cnn.initializeWeightsGaussian(sigma);
                initFunc = @(x) cnn.initializeWeights;
                optFunc = @(T,stop_cond) cnn.Nesterov(T,@(t)(res.base_lr),...
                                                        res.mu,lam,printIter,projection,0,[],[],stop_cond);
                tic;
                [res.init_theta, res.AllLossNes,res.seed] = ...
                    cnn.OptimizationWrapper(initFunc,optFunc,T_trial,threshold,limit,T);
                toc
                if length(res.AllLossNes)==1
                    res.AllLossNes = NaN*ones(2,T/printIter + 1);
                end
                res.final_theta = cnn.theta;
                if isnan(res.AllLossNes(end)) %||res.AllLossNes(end)> 0.2 
                    res.final_loss = [0,0];
                else
                    res.final_loss = cnn.calcLossAndErr();
                end
                testNet.setTheta(cnn.theta);
                res.TestLoss = testNet.calcLossAndErr();
                fprintf(1,'Train loss= %f, Train accuracy = %f\n',res.final_loss(1),1-res.final_loss(2));
                fprintf(1,'Test loss= %f, Test accuracy = %f\n',res.TestLoss(1),1-res.TestLoss(2));
                results{end+1} = res;
                
                [fiel] = createCSV(results,'demo.csv');
                end


            end
        end
% end


%%
% calculate test error

testlenet = lenet;
testlenet{1}.fName = fullfile(dataLocation ,'test.images.bin');
testlenet{2}.fName = fullfile(dataLocation, 'test.labels.bin');
testNet = ConvNet(testlenet,atGPU);
testNet.setTheta(cnn.theta);
TestLoss = testNet.calcLossAndErr();
fprintf(1,'Test loss= %f, Test accuracy = %f\n',TestLoss(1),1-TestLoss(2));

