function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));


Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) ,  X];
cost = 0;
temp = 1:num_labels;
Delta2 = zeros(size(Theta2_grad));
Delta1 = zeros(size(Theta1_grad));
%size(Delta1)
%size(Delta2)

T1_l = [zeros(size(Theta1),1),ones(size(Theta1,1),size(Theta1,2)-1)];
T2_l = [zeros(size(Theta2),1),ones(size(Theta2,1),size(Theta2,2)-1)];

for i = 1:m,
	x = X(i,:)';
	z2 = Theta1 * x;
	a2 = sigmoid(z2);
	a2 = [1;a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	h = a3;
	yi = (temp == y(i))';
	s = yi.*log(h) + (1-yi).*log(1-h);
	cost = cost - 1*sum(s)/m;
	% implementing back prop
	z2 = [1;z2];
	delta3 = h - yi;
	delta2 = (Theta2'*delta3).*sigmoidGradient(z2);
	delta2(1,:) = [];
	%size(delta3)
	%size(delta2)

	Delta2 = Delta2 + delta3*a2';
	Delta1 = Delta1 + delta2*x';

	l2 = [0;ones(size(Theta2,1)-1,1)];
	l1 = [0;ones(size(Theta1,1)-1,1)];

	Theta1_grad = (lambda*Theta1.*T1_l + Delta1)/m;
	Theta2_grad = (lambda*Theta2.*T2_l + Delta2)/m;
end;


Theta1_sum = sum(sum((Theta1.*T1_l).^2));
Theta2_sum = sum(sum((Theta2.*T2_l).^2));


J = cost;

% Implementing regularization
J = J + lambda * (Theta1_sum + Theta2_sum )/(2*m);


	

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
