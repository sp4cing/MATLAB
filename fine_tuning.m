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

% reading sample image
I = read(imds);

figure,imshow(I,[])
%% Splitting Dataset train and test

trn_ratio = 0.5;
[trn_set, tst_set] = splitEachLabel(imds , trn_ratio, 'randomize');

%% Building Fine-tuning Network

% pre-trained net store: 
% googlenet inceptionv3 mobilenetv2 resnet18 resnet50 xception  
% https://ch.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html
% It is noted that you should install pre trained networks from Apps before
% using below codes 

net = resnet18;
net.Layers(1)

% @single_image_preprocess_finetuning is important for making suitable image size
% with network input size. Hence, we should learn network input size from "net.Layers(1)"  
% In this example it is [224 224 3]

trn_set.ReadFcn = @single_image_preprocess_finetuning;
tst_set.ReadFcn = @single_image_preprocess_finetuning;

%%
%
if isa(net,'SeriesNetwork') 
  cnn_layers = layerGraph(net.Layers); 
else
  cnn_layers = layerGraph(net);
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% THESE CODES ARE FROM: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% https://ch.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[learnableLayer,classLayer] = findLayersToReplace(cnn_layers);

% In our project, it is 5
numClasses = 5;

% WeightLearnRateFactor and BiasLearnRateFactor can be changed
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

cnn_layers = replaceLayer(cnn_layers,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
 
cnn_layers = replaceLayer(cnn_layers,classLayer.Name,newClassLayer);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

% For resnet18, feature can be extracted from  "pool5".
 
feat_layer = 'pool5';

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


