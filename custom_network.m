%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   ____  ____   _    ____ ___ _   _  ____      %
%  / ___||  _ \ / \  / ___|_ _| \ | |/ ___|     %
%  \___ \| |_) / _ \| |    | ||  \| | |  _      %
%   ___) |  __/ ___ \ |___ | || |\  | |_| |     %
%  |____/|_| /_/   \_\____|___|_| \_|\____|     %
%                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% https://spacing.itu.edu.tr/

%% system initialization 
clear all, close all; clc

% This command deletes all windows in Matlab even if it is DL training plots 
% delete(findall(0));

% rng() is used to generate the same random numbers each time the program is run.
rng(0) 

 
data_base = 'data_set';
%% Preparing image data store 

% With imageDatastore, the urls of the images are stored from all folders 
% under the data set folder, with the name of the folder being the class name.
% Storing image urls is the most effective way to work with large amounts of data set.
% " 'FileExtensions', {'.jpg', '.tif'} " can be added to read images in certain formats from folders.

imds = imageDatastore(data_base, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

% If there are some pre-processing operations to be performed after the image 
% is read, these procedures can be written into the function file. In this way, 
% when the pictures are read before they are processed, they are given to the 
% network after pre-processing.

imds.ReadFcn = @single_image_preprocess_custom;

% reading sample image
I = read(imds);

figure,imshow(I,[])
%% Splitting Dataset train and test

trn_ratio = 0.5;
[trn_set, tst_set] = splitEachLabel(imds , trn_ratio, 'randomize');

%% Building Custom Network

% basic_conv includes:
% convolution2dLayer
% batchNormalizationLayer
% activation

% basic_conv( filter_size , filter_number , which_activation , stride )

% You can change normalization function in imageInputLayer 
% Normalization — Data normalization
% 'zerocenter' (default) | 'zscore' | 'rescale-symmetric' | 'rescale-zero-one' |

% Activations
% reluLayer();
% leakyReluLayer(0.01)
% tanhLayer()

% Pooling
% averagePooling2dLayer
% maxPooling2dLayer

tmp_act = reluLayer();

cnn_layers = [
    imageInputLayer([64 , 64 , 3] ,'Normalization' , 'zerocenter')
    
    
    basic_conv( [5 5] , 32 , tmp_act , [2 2])                              % 32 32
    maxPooling2dLayer([3 3 ], 'Stride', [2 2], 'Padding' , 'same')         % 16 16
    
    
    basic_conv( [3 3] , 64 , tmp_act , [1 1])
    maxPooling2dLayer([3 3 ], 'Stride', [2 2], 'Padding' , 'same')         % 8 8
    
    
    basic_conv( [3 3] , 128 , tmp_act , [1 1])
    maxPooling2dLayer([3 3 ], 'Stride', [2 2], 'Padding' , 'same')         % 4 4
    
   
    basic_conv( [3 3] , 256 , tmp_act , [1 1])
    maxPooling2dLayer([3 3 ], 'Stride', [2 2], 'Padding' , 'same')         % 2 2
    
    globalAveragePooling2dLayer                                            % 1 1
    batchNormalizationLayer
    tmp_act
    
    fullyConnectedLayer(512)
    
    dropoutLayer(0.5)
    
    fullyConnectedLayer(5) % this must be number of class
    
    softmaxLayer
    
    classificationLayer
    
    ];


analyzeNetwork(cnn_layers)
%%


options = trainingOptions(...
    'adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',100, ...
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch',...
    'ValidationData',tst_set,...
    'ValidationFrequency', 50 , ...
    'ValidationPatience' ,  Inf , ...
    'Plots','training-progress',...
    'Verbose',0);


[net , info ] = trainNetwork(trn_set , cnn_layers,options);
%% Classification of test images

% Please write your own code for test image classification.
% https://ch.mathworks.com/help/deeplearning/ref/classify.html
% It is important that labels are NOT numerical, they are categorical

%% Using CNN as Feature Extractor 

% After processing an image via CNN, a feature vector can be obtained for 
% classification in handcrafted classifiers such as SVM, kNN, ect. 

% Our 1x1xN feature generating layers: gap and fc_1
% fc_1 is selected for getting feature vectors. 

% gap 
% fc_1

feat_layer = 'fc_1';

trn_feat = activations(net,trn_set,feat_layer,'OutputAs','rows');
tst_feat = activations(net,tst_set,feat_layer,'OutputAs','rows');
trn_gnd  = trn_set.Labels;
tst_gnd  = tst_set.Labels;


temp_learner = templateSVM(...
    'KernelFunction','rbf',...
    'KernelScale','auto',...
    'BoxConstraint',1,...
    'Standardize',0 );

Mdl = fitcecoc(trn_feat,trn_gnd,'Learners',temp_learner);

[pre_lbl,pre_scr] = predict(Mdl,tst_feat);

acc = 100 * ( 1 - nnz(double(pre_lbl)-double(tst_gnd))/numel(tst_gnd) )

cmat = confusionmat(tst_gnd,pre_lbl);
figure,heatmap(cmat)
%% kNN Classification

% Please write kNN classifier code by using:
% https://ch.mathworks.com/help/stats/classificationknn.html
