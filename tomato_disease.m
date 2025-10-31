% Path to dataset
datasetPath = 'C:\Users\B Gautam\OneDrive\Attachments\Documents\MATLAB\PlantDiseaseDataset';

% Create image datastores for training and validation
imdsTrain = imageDatastore(fullfile(datasetPath,'train'), ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsVal = imageDatastore(fullfile(datasetPath,'validation'), ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Resize images to match ResNet-50 input
inputSize = [224 224 3];
imdsTrain.ReadFcn = @(x) imresize(imread(x), inputSize(1:2));
imdsVal.ReadFcn   = @(x) imresize(imread(x), inputSize(1:2));

% Load pretrained ResNet-50
net = resnet50;
lgraph = layerGraph(net);

% Find number of classes in your dataset
numClasses = numel(categories(imdsTrain.Labels));

% Define new final layers
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fcPlantDisease', ...
        'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmaxPlantDisease')
    classificationLayer('Name','classOutput')];

% Replace last layers of ResNet-50
lgraph = replaceLayer(lgraph,'fc1000',newLayers(1));
lgraph = replaceLayer(lgraph,'fc1000_softmax',newLayers(2));
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newLayers(3));

% Freeze all layers except the last 3
layers = lgraph.Layers;
connections = lgraph.Connections;

for i = 1:numel(layers)-3
    if isprop(layers(i),'WeightLearnRateFactor')
        layers(i).WeightLearnRateFactor = 0;
    end
    if isprop(layers(i),'BiasLearnRateFactor')
        layers(i).BiasLearnRateFactor = 0;
    end
end

% Rebuild layer graph with frozen layers + preserved skip connections
lgraph = createLgraphUsingConnections(layers,connections);

% Training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsVal, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','gpu');

% Train the network and capture training info
[netTransfer, trainInfo] = trainNetwork(imdsTrain,lgraph,options);

% Save trained model
save('trainedCNN.mat','netTransfer','trainInfo');

% Plot and save Training Accuracy
figure;
plot(trainInfo.TrainingAccuracy,'LineWidth',2);
xlabel('Epoch'); ylabel('Training Accuracy (%)');
title('Training Accuracy');
grid on;
saveas(gcf,'training_accuracy.png');
savefig('training_accuracy.fig');

% Plot and save Training Loss
figure;
plot(trainInfo.TrainingLoss,'LineWidth',2);
xlabel('Epoch'); ylabel('Training Loss');
title('Training Loss');
grid on;
saveas(gcf,'training_loss.png');
savefig('training_loss.fig');

% Evaluate validation accuracy
[YPred,probs] = classify(netTransfer, imdsVal);
YValidation = imdsVal.Labels;
accuracy = mean(YPred == YValidation);
disp("Validation accuracy: " + accuracy*100 + "%");

% Confusion matrix
figure;
cm = confusionchart(YValidation, YPred);
cm.Title = 'Confusion Matrix';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
saveas(gcf,'confusion_matrix.png');
savefig('confusion_matrix.fig');



% -----------------------------
% Helper function (keep at end)
% -----------------------------
function lgraph = createLgraphUsingConnections(layers,connections)
    lgraph = layerGraph();
    for i = 1:numel(layers)
        lgraph = addLayers(lgraph,layers(i));
    end
    for c = 1:size(connections,1)
        lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
    end
end
