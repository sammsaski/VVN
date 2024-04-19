train_data_filepath = "/home/sasakis/git/VVN/src/datasets/matlab/mnistvideo_zoom_in_32x32_train_data_seq.mat";
train_labels_filepath = "/home/sasakis/git/VVN/src/datasets/matlab/mnistvideo_zoom_in_32x32_train_labels_seq.mat";
test_data_filepath = "/home/sasakis/git/VVN/src/datasets/matlab/mnistvideo_zoom_in_32x32_test_data_seq.mat";
test_labels_filepath = "/home/sasakis/git/VVN/src/datasets/matlab/mnistvideo_zoom_in_32x32_test_labels_seq.mat";

sequences = load(train_data_filepath);
labels = load(train_labels_filepath);

% samples = test(1);
% disp(size(samples.data));
% 
% samples_labels = test_labels(1);
% disp(size(samples_labels.data));

numObservations = numel(sequences);
idx = randperm(numObservations);
N = floor(0.9 * numObservations);

idxTrain = idx(1:N);
sequencesTrain = sequences(idxTrain);
labelsTrain = labels(idxTrain);

idxValidation = idx(N+1:end);
sequencesValidation = sequences(idxValidation);
labelsValidation = sequences(idxValidation);


layers = [
    % video input
    sequenceInputLayer([32 32 1]);

    % Layer 1
    convolution3dLayer(3, 64, "Padding", 1);
    reluLayer();
    averagePooling3dLayer([1 1 2], "Stride", 2);

    % Layer 2
    convolution3dLayer(3, 128, "Padding", 1);
    reluLayer();
    averagePooling3dLayer(2, "Stride", 2);

    % Layer 3
    convolution3dLayer(3, 256, "Padding", 1);
    reluLayer();
    convolution3dLayer(3, 256, "Padding", 1);
    reluLayer();
    averagePooling3dLayer(2, "Stride", 2);

    % Layer 4
    convolution3dLayer(3, 512, "Padding", 1);
    reluLayer();
    convolution3dLayer(3, 512, "Padding", 1);
    reluLayer();
    averagePooling3dLayer(2, "Stride", 2);

    % Layer 5
    convolution3dLayer(3, 512, "Padding", 1);
    reluLayer();
    convolution3dLayer(3, 512, "Padding", 1);
    reluLayer();
    averagePooling3dLayer(2, "Stride", 2, "Padding", [0 1 1]);

    % flatten before fully-connected layers
%     flattenLayer()

    % Layer 6 (fully-connected)
    fullyConnectedLayer(1024, "Name", "fc6");
    reluLayer();
    dropoutLayer(0.5);

    % layer 7
    fullyConnectedLayer(1024, "Name", "fc7");
    reluLayer();
    dropoutLayer(0.5);

    % layer 8
    fullyConnectedLayer(10, "Name", "fc8"); % num classes
];

options = trainingOptions("adam", ...
    'InitialLearnRate', 1e-3 / 255, ...
    'Plots', 'training-progress', ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 16);

[c3dNet, info] = trainNetwork(sequencesTrain, labelsTrain, layers, options);

YPred = classify(c3dNet, sequencesValidation, 'MiniBatchSize', 16);
YValidation = labelsValidation;
accuracy = mean(YPred == YValidation);