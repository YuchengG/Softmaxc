%% Softmax Exercise by guoyucheng
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


display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));

inputData = images;

% For debugging purposes, we may wish to reduce the size of the input data
% in order to speed up gradient checking. 
% Here, we create synthetic dataset using random data for testing

DEBUG = false; % Set DEBUG to true when debugging.
if DEBUG
    inputSize = 8;
    inputData = randn(8, 100); %Normally distributed random numbers and arrays 8*100
    labels = randi(10, 100, 1); %100*1 matrix, and the number is between 0 and 10
end

% Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize, 1);

%%======================================================================
%% STEP 2: Implement softmaxCost
%
%  Implement softmaxCost in softmaxCost.m. 

[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
                                     
%%======================================================================
%% STEP 3: Gradient checking
%
%  As with any learning algorithm, we should always check that our
%  gradients are correct before learning the parameters.
% 

if DEBUG
    numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
                                    inputSize, lambda, inputData, labels), theta);

    % Use this to visually compare the gradients side by side
    disp([numGrad grad]); 

    % Compare numerically computed gradients with those computed analytically
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
    % The difference should be small. 
    % In our implementation, these values are usually less than 1e-7.

end

%%======================================================================
%% STEP 4: Learning parameters
%
%  Once we have verified that our gradients are correct, 
%  we can start training our softmax regression code using softmaxTrain
%  (which uses minFunc).
% Randomly initialise theta StochGD softmaxTrain

options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);
                          
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

display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));

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
