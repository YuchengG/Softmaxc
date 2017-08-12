function [softmaxModel] = StochGD(inputSize, numClasses, lambda, inputData, labels, options)
%softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
%
% inputSize: the size of an input vector x^(i)
% numClasses: the number of classes 
% lambda: weight decay parameter
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input
% options (optional): options
%   options.maxIter: number of iterations to train for

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 400;
end

% initialize parameters
theta = 0.005 * randn(numClasses * inputSize, 1);
cost_history = zeros(options.maxIter + 1,1);
% Use stochastic Gradient Descent to minimize the function
number = size(inputData,2); 
[cost_history(1), ~]= softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
   
for j = 1 : options.maxIter
       index = randperm(number); 
       for i = 1 : number  
          [theta, cost] = StochGraDes( @(p) softmaxCost(p,numClasses,inputSize,lambda,inputData(:,index(i)),labels(index(i))),theta, options);
       end
       [cost_history(j + 1), ~]= softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels); 
       
       if ((cost_history(j + 1) - cost_history(j))^2< 1e-10) 
           break;
       end
end
    softmaxOptTheta = theta;

% Fold softmaxOptTheta into a nicer format
softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;
softmaxModel.cost_history = cost_history;
                          
end  


