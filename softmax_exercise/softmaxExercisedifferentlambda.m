%% Softmax Exercise by guoyucheng
%Time:2013/12/18
%%======================================================================
%% STEP 0: Initialise constants and parameters
%
clc;
clear;
inputSize = 28 * 28; % Size of input vector (MNIST images are 28x28)
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)

lambdaall = [7.5e-6 ,5e-6,2.5e-6,1e-6, 7.5e-7 ,5e-7,2.5e-7,1e-7, 7.5e-8 ,5e-8,2.5e-8,1e-8, 7.5e-9 ,5e-9,2.5e-9,1e-9, 7.5e-10 ,5e-10,2.5e-10,1e-10];
trainacc = zeros(length(lambdaall),2);

%train images
trainimages = loadMNISTImages('MNIST_database/train-images.idx3-ubyte');
trainlabels = loadMNISTLabels('MNIST_database/train-labels.idx1-ubyte');
%test images
testimages = loadMNISTImages('MNIST_database/t10k-images.idx3-ubyte');
testlabels = loadMNISTLabels('MNIST_database/t10k-labels.idx1-ubyte');

for ij = 1:length(lambdaall)
%lambda = 1e-4; % Weight decay parameter
lambda = lambdaall(ij);
%%======================================================================
%% STEP 1: Load data
%
%  For softmax regression on MNIST pixels, 
%  the input data is the images, and 
%  the output data is the labels.
%
%tic;
%toc;

trainlabels(trainlabels==0) = 10; % Remap 0 to 10


inputData = trainimages;

%%======================================================================
%% STEP 4: Learning parameters
%
%  Once we have verified that our gradients are correct, 
%  we can start training our softmax regression code using softmaxTrain
%  (which uses minFunc).
% Randomly initialise theta

options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, trainlabels, options);
                          
% Although we only use 100 iterations here to train a classifier for the 
% MNIST data set, in practice, training for more iterations is usually
% beneficial.

%%======================================================================
%% STEP 5: Testing
%
%  We should now test our model against the test images.
%  To do this, we will first need to write softmaxPredict
%  (in softmaxPredict.m), which should return predictions
%  given a softmax model and the input data.
% we will have to implement softmaxPredict in softmaxPredict.m

[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(trainlabels(:) == pred(:));
%fprintf('Accuracy: %0.3f%%\n', acc * 100);
trainacc(ij,1) = acc;

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

testlabels(testlabels==0) = 10; % Remap 0 to 10

inputData = testimages;

% we will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(testlabels(:) == pred(:));
%fprintf('Accuracy: %0.3f%%\n', acc * 100);
trainacc(ij,2) = acc;
% Accuracy is the proportion of correctly classified images
% After 100 iterations, the results for our implementation were:
%
% Accuracy: 92.200%
%
end