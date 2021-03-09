function Lab5
data = load('wine.txt'); %tae dataset from UCI
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

dimensionTrain = size(xTrain,2);
dimensionTest = size(tTrain,2);
n = 0.01;
hNode = hidden_node*10; %Hidden Node
wInput = rands(dimensionTrain,hNode);
biasInput = rands(1,hNode);
wOutput = rands(hNode,dimensionTest);
biasOutput = rands(1,dimensionTest);
E = [];

%MLP
for k = 1:50
    for i = 1:size(xTrain,1)
        H = logsig(xTrain(i,:)*wInput + biasInput);
        Y = logsig(H*wOutput + biasOutput);
        
        e = tTrain(i,:) - Y;
        
        dy = e .* Y.*(1-Y);
        dH = H.*(1-H) .* (dy*wOutput');
        
        wOutput = wOutput + n * H'*dy;
        biasOutput = biasOutput + n * dy;
        wInput = wInput + n * xTrain(i,:)'*dH;
        biasInput = biasInput + n * dH;
    end
    H = logsig(xTrain*wInput + repmat(biasInput,size(xTrain,1),1));
    Y = logsig(H*wOutput + repmat(biasOutput,size(xTrain,1),1));
    %E(k) = mse(tTrain - Y);

    %disp("T : " + k + " Error : " + E(k));
end

HTrain = logsig(xTrain*wInput + repmat(biasInput,size(xTrain,1),1));
YTrain = logsig(HTrain*wOutput + repmat(biasOutput,size(xTrain,1),1));
HTest = logsig(xTest*wInput + repmat(biasInput,size(xTest,1),1));
YTest = logsig(HTest*wOutput + repmat(biasOutput,size(xTest,1),1));

[temp,indexTrain1] = max(YTrain,[],2);
[temp,indexTrain2] = max(tTrain,[],2);
%fprintf('Training acc. : %f\n',mean(mean(indexTrain1 == indexTrain2))*100);

[temp,indexTest1] = max(YTest,[],2);
[temp,indexTest2] = max(tTest,[],2);
%fprintf('Testing acc. : %f\n',mean(mean(indexTest1 == indexTest2))*100);

training_case(round) = mean(mean(indexTrain1 == indexTrain2))*100;
testing_case(round) = mean(mean(indexTest1 == indexTest2))*100;

        end %testing case
fprintf('Model hidden node: %d\n',hNode);      
fprintf('Avg training: %f\n',mean(training_case));
fprintf('Avg testing: %f\n',mean(testing_case));
fprintf('---------------------------------------\n');
    end
end