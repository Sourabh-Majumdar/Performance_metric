% data pre processing step
raw_data_temp = load('data_banknote_authentication.txt');
raw_data = raw_data_temp(randperm(size(raw_data_temp,1)),:);
n = size(raw_data,2) - 1;
X = raw_data(:,1:n);
y = raw_data(:,n+1);
m = size(X,1)
X_train = X(1:ceil(0.6*m),:)
y_train = y(1:ceil(0.6*m),1)

X_val = X(ceil(0.6*m)+1:ceil(0.8*m),:)
y_val = y(ceil(0.6*m)+1:ceil(0.8*m),1)

X_test = X(ceil(0.8*m)+1:m,:)
y_test = y(ceil(0.8*m)+1:m,1)

% Train SVM Classifier

x1 = [1 2 1];
x2 = [0 4 -1]; 

[C, sigma] = dataset3Params(X_train, y_train, X_val, y_val);

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

predictions = svmPredict(model,X_test);

accuracy_1 = sum((predictions == 1)&(y_test == 1));
accuracy_2 = sum((predictions == 0)&(y_test == 0));
accuracy = (accuracy_1 + accuracy_2) / size(y_test,1)
fprintf('\nTest Set Accuracy: %f\n', mean(double(predictions == y_test)) * 100);



