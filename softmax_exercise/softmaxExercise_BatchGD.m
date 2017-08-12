%% Softmax Exercise by guoyucheng
%Batch Gradient Descent
%Time:2013/12/18
%%======================================================================
%% STEP 0: Initialise constants and parameters
%
clc;
clear;
inputSize = 28 * 28; % Size of input vector (MNIST images are 28x28)
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)

lambda = 1e-4; % Weight decay parameter

%%======================================================================
%% STEP 1: Load data
%
%  For softmax regression on MNIST pixels, 
%  the input data is the images, and 
%  the output data is the labels.
%


images = loadMNISTImages('MNIST_database/train-images.idx3-ubyte');
labels = loadMNISTLabels('MNIST_database/train-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10


% display_network
%display_network(images(:,1:100)); % Show the first 100 images
%disp(labels(1:10));

inputData = images;

% Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize, 1);


%%======================================================================
%% STEP 2: Learning parameters by Batch Gradient Descent
%Batch Gradient Descent
% Randomly initialise theta

options = 500;
tic;
softmaxModel = batchGD(inputSize, numClasses, lambda, inputData, labels, options);
toc
costnt = softmaxModel.costnt;

[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);
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

images = loadMNISTImages('MNIST_database/t10k-images.idx3-ubyte');
labels = loadMNISTLabels('MNIST_database/t10k-labels.idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%display_network(images(:,1:100)); % Show the first 100 images
%disp(labels(1:10));

inputData = images;

% we will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% After 100 iterations, the results for our implementation were:
%
% Accuracy: 92.200%
%
plot(costnt(:,1),costnt(:,2));
title('The cost for the count ');
xlabel('The number of the iterations');
ylabel('The cost values');