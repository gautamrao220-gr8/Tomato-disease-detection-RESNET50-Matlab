% -----------------------------
% Load trained CNN
% -----------------------------
load('C:\Users\B Gautam\OneDrive\Desktop\CNN model for plant_disease classification\tomato\model.mat','netTransfer'); % load the trained network

% -----------------------------
% Path to test image
% -----------------------------
testImagePath = 'C:\Users\B Gautam\OneDrive\Desktop\CNN model for plant_disease classification\tomato\PlantDiseaseDataset\validation\Tomato___Leaf_Mold\0ba88812-fa4f-4602-bcec-c03cd7d0ba2b___Crnl_L.Mold 6990.JPG'; % adjust as needed

% Read and resize image to match network input
img = imread(testImagePath);
imgResized = imresize(img, [224 224]); % ResNet-50 input size

% -----------------------------
% Classify image
% -----------------------------
[label, scores] = classify(netTransfer, imgResized);

% Display result
imshow(img);
title(sprintf('Predicted: %s, Probability: %.2f%%', string(label), max(scores)*100));

% -----------------------------
% Optional: Display top 5 predictions
% -----------------------------
[sortedScores, idx] = sort(scores,'descend');
categories = netTransfer.Layers(end).Classes;

fprintf('Top 5 Predictions:\n');
for i = 1:min(5,numel(sortedScores))
    fprintf('%s: %.2f%%\n', string(categories(idx(i))), sortedScores(i)*100);
end
