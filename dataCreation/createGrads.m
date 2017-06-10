%% Train images
fName = '/mnt/local-ssd/nitzak01/simcep/exp2/cells_s_150_N_50_p_0_2.bin';
dataType = 'int8';
blobSize = [150,150,3,100];
scale = 1/127;
maxExamples = 100;
atGPU = 0;
dataFormat = 'bin';
comp = 0;
D = dataClass(fName,dataType,blobSize,scale,maxExamples,atGPU,dataFormat,comp);
%%
output_filename =  '/mnt/local-ssd/nitzak01/simcep/simple/exp1/cells_s_150_N_50_p_0_2_grads_fixed.bin';
output_fid = fopen(output_filename,'ab'); %Does not override! just appends.
output_data_type = 'uint8';
convert_input_fixed(D,output_fid,output_data_type);
fclose(output_fid);


%% Test set
fName_test = '/mnt/local-ssd/nitzak01/simcep/exp2/cells_s_150_N_50_p_0_2_test.bin';
D_test = dataClass(fName_test,dataType,blobSize,scale,maxExamples,atGPU,dataFormat,comp);
output_filename_test =  '/mnt/local-ssd/nitzak01/simcep/simple/exp1/cells_s_150_N_50_p_0_2_grads_fixed_test.bin';
output_fid_test = fopen(output_filename_test,'ab'); %Does not override! just appends.
convert_input_fixed(D_test,output_fid_test,output_data_type);
fclose(output_fid_test);




%% Process the labels set
fName =  '/mnt/local-ssd/nitzak01/simcep/exp2/gt_s_150_N_50_p_0_2.bin';
dataType = 'int8';
blobSize = [150,150,3,100];
maxExamples = 100;
scale = 1;
dataFormat = 'bin';
comp = 0;
atGPU = 0;
D_labels = dataClass(fName,dataType,blobSize,scale,maxExamples,atGPU,dataFormat,comp);


%%
output_filename =  '/mnt/local-ssd/nitzak01/simcep/simple/exp1/gt_boolean_exp1_thresh_10.bin';
output_fid = fopen(output_filename,'ab');
output_data_type = 'int8';
convert_gt(D_labels,output_fid,output_data_type);
fclose(output_fid);


%% Test set
fName_test = '/mnt/local-ssd/nitzak01/simcep/exp2/gt_s_150_N_50_p_0_2_test.bin';
D_labels_test = dataClass(fName_test,dataType,blobSize,scale,maxExamples,atGPU,dataFormat,comp);
output_filename_test =  '/mnt/local-ssd/nitzak01/simcep/simple/exp1/gt_boolean_exp1_thresh_10_test.bin';
output_fid_test = fopen(output_filename_test,'ab'); %Does not override! just appends.
convert_gt(D_labels_test,output_fid_test,output_data_type);
fclose(output_fid_test);



