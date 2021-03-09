function projectELM
%Classification
data = load('wine.txt'); %wine dataset from UCI
X = mapminmax(data(:,2:end)',0,1)'; %Normalize data
T = data(:,1);
data_size = size(X,1);
round_data = floor(data_size*0.7);

targetTemp = [];
index = 1;
for i = 1:size(T,1)
    if ismember(T(i),targetTemp) == 0
        targetTemp(index) = T(i);
        index = index + 1;
    end
end

targetValue = [];
for i = 1:size(T,1)
    for j = 1:index-1
        if T(i) == targetTemp(j)
            targetValue(i,:) = zeros(1,size(targetTemp,2));
            targetValue(i,j) = 1;
            break
        end
    end
end

for hidden_node = 3:7

training_case = [];
testing_case = [];

for round = 1:50

I = randperm(data_size);
xTrain = X(I(1:round_data),:);
tTrain = targetValue(I(1:round_data),:);
xTest = X(I(round_data+1:end),:);
tTest = targetValue(I(round_data+1:end),:);

dimension = size(xTrain,2);
hNode = hidden_node*10;

wInput = unifrnd(0,1,dimension,hNode);
bias = unifrnd(0,1,1,hNode);

hLayer = 1./(1+exp(-xTrain*wInput+repmat(bias,size(xTrain,1),1)));
wOutput = pinv(hLayer)*tTrain;
yTrain = hLayer*wOutput;

hLayer = 1./(1+exp(-xTest*wInput+repmat(bias,size(xTest,1),1)));
yTest = hLayer*wOutput;

%fprintf('Training Output : %e \n',mse(tTrain-yTrain));
%fprintf('Training Output : %e \n',mse(tTest-yTest));

%hold on
%plot();
%legend('Target Output','Output of Network');

[temp,indexTrain1] = max(yTrain,[],2);
[temp,indexTrain2] = max(tTrain,[],2);
%fprintf('Training acc. : %f\n',mean(mean(indexTrain1 == indexTrain2))*100);

[temp,indexTest1] = max(yTest,[],2);
[temp,indexTest2] = max(tTest,[],2);
%fprintf('Testing acc. : %f\n',mean(mean(indexTest1 == indexTest2))*100);


training_case(round) = mean(mean(indexTrain1 == indexTrain2))*100;
testing_case(round) = mean(mean(indexTest1 == indexTest2))*100;
tempround(hidden_node-2) = hNode;


        end %testing case
fprintf('Model hidden node: %d\n',hNode);      
fprintf('Avg training: %f\n',mean(training_case));
fprintf('Avg testing: %f\n',mean(testing_case));
fprintf('---------------------------------------\n');
    end
end %node