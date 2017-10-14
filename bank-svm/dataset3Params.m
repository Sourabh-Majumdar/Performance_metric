function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
x1 = [1,2,1];
x2 = [0,4,-1];
z = [0.01,0.03,0.1,0.3,1,3,10,30];
c_min = 0.01;
sig_min = 0.01;

model= svmTrain(X, y, c_min, @(x1, x2) gaussianKernel(x1, x2,sig_min));
 
predictions = svmPredict(model, Xval);
err_min = mean(double(predictions ~= yval));

for c_i = z,
	for sig_i = z,
		model= svmTrain(X, y, c_i, @(x1, x2) gaussianKernel(x1, x2, sig_i)); 
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));
		if (err < err_min),
			%err
			%err_min
			c_min = c_i;
			sig_min = sig_i;
			err_min = err;
		end
	end
end
C = c_min;
sigma = sig_min;


% =========================================================================

end
