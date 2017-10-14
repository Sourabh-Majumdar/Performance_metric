% loading and cleaning of parameters

temp = load('data_banknote_authentication.txt')
raw_data = temp(randperm(size(temp,1)),:)

n = size(raw_data,2) - 1;
m = size(raw_data,1);

X = raw_data(:,1:n);
y = raw_data(:,n+1);

X_train = X(1:ceil(0.7*m),:)
y_train = y(1:ceil(0.7*m),1)

%X_val = X(ceil(0.6*m)+1:ceil(0.8*m),:)
%y_val = y(ceil(0.6*m)+1:ceil(0.8*m),1)

X_test = X(ceil(0.7*m)+1:m,:)
y_test = y(ceil(0.7*m)+1:m,1)

% Neural-Net Classifier

input_layer_size = n;
hidden_layer_size = 16;
num_labels = 1;

Theta1 = rand(16,5);
Theta2 = rand(1,17);

lambda = 0;

nn_params = [Theta1(:);Theta2(:)];

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

ini_Theta1 = rand(16,5);
ini_Theta2 = rand(1,17);

initial_nn_params = [ini_Theta1(:);ini_Theta2(:)];

options = optimset('MaxIter', 1000);
lambda = 1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X_test);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

