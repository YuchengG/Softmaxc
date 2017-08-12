function [softmaxModel] = batchGD(inputSize, numClasses, lambda, inputData, labels, options)
    %inputSize, numClasses, lambda, inputData, labels, options
    %define the update rate of the parameter
    alpha = 0.8;
    %set old cost and new cost value
    cost_old = 0;
    cost = 1;
    %count the running number
    count = 0;
    % Randomly initialise theta
    costnt = zeros(options,2);
    theta = 0.005 * randn(numClasses * inputSize, 1);
    %Learning parameters by Batch Gradient Descent
    while (( abs(cost_old - cost) > 0.0001*cost ) & count < options )
        cost_old = cost;
        [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
        theta = theta - alpha*grad;
        count = count+1;
        %show the cost and count
        costnt(count,1) = count;
        costnt(count,2) = cost;
        %display([cost,count]);
    end
    % Fold softmaxOptTheta into a nicer format
    softmaxModel.optTheta = reshape(theta, numClasses, inputSize);
    softmaxModel.inputSize = inputSize;
    softmaxModel.numClasses = numClasses;
    softmaxModel.costnt = costnt;
end