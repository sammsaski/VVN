models = "../models/bcss_model_op11_v1.onnx";

netonnx = importONNXNetwork(models, "InputDataformats", "BCSS", "OutputDataFormats", "BC");
net = matlab2nnv(netonnx);

disp("finishedl oading model.");

data = readNPY('../datasets/numpy/mnistvideo_zoom_out_32x32_test_data_seq.npy');
label = readNPY('../datasets/numpy/mnistvideo_zoom_out_32x32_test_labels_seq.npy');

data_squeezed = permute(squeeze(data), [3, 4, 2, 1]);
test = data_squeezed(:, :, :, :);

% disp(test(:, :, :, 1));

output = predict(netonnx, test);
tot_incorrect = 0;
[~, predictedLabels] = max(output, [], 2);

for i=1:1000
    if predictedLabels(i) ~= label(i)+1
        tot_incorrect = tot_incorrect + 1;
    end
end

disp(1 - (tot_incorrect / 1000));

