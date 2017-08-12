function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% our code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- SoftmaxPredict --------------------------------------
%  Compute pred using theta assuming that the labels start from 1.
M = bsxfun(@minus,theta*data,max(theta*data, [], 1));
M = exp(M);
predall = bsxfun(@rdivide, M, sum(M));

[probablity ,pred] = max(predall);


% ---------------------------------------------------------------------

end

