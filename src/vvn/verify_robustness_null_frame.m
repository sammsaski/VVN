disp("Running robustness verification...");

if ~exist("./results/null_frame", 'dir')
    mkdir("results/null_frame");
end

% Load the model
models = "../models/bcss_model_op11_v1.onnx";
netonnx = importONNXNetwork(models, "InputDataformats", "BCSS", "OutputDataFormats", "BC");
net = matlab2nnv(netonnx);
disp("finished loading model.");

% Load the dataset
data = readNPY('../datasets/numpy/mnistvideo_zoom_out_32x32_test_data_seq.npy');
labels = readNPY('../datasets/numpy/mnistvideo_zoom_out_32x32_test_labels_seq.npy');
data_squeezed = permute(squeeze(data), [3, 4, 2, 1]);
test = data_squeezed(:, :, :, :);

N = 10; % Number of images to evaluate
nR = 297; % Number of images to sample

% Define reachability options
reachOptions.reachMethod = 'relax-star-area';
reachOptions.relaxFactor = 0.5;

% Size of attack
epsilon = 1/255;

% Results
res = zeros(N, 1);
time = zeros(N, 1);
met = repmat("relax", N, 1);

for i=1:N
    fprintf("Iteration number : %d \n", i);
    [IS, xRand] = L_inf_attack(test(:,:,:,i), epsilon, nR, 10);
    t = tic;
    predictedLabels = predict(netonnx, xRand);
    [~, predictedLabels] = max(predictedLabels, [], 2);
    if any(predictedLabels-1 ~= labels(i))
        res(i) = 0;
        time(i) = toc(t);
        met(i) = "counterexample";
        continue;
    end

    try
        temp = net.verify_robustness(IS, reachOptions, labels(i)+1);
        if temp ~= 1 && temp ~= 0
            reachOptions = struct;
            reachOptions.reachMethod = 'approx-star';
            temp = net.verify_robustness(IS, reachOptions, labels(i)+1);
            met(i) = 'approx';
        end
    catch ME
        met(i) = ME.message;
        temp = -1;
    end

    res(i) = temp;
    time(i) = toc(t);

    reachOptions.reachMethod = 'relax-star-area';
    reachOptions.relaxFactor = 0.5;
end

% Results MAKE SURE TO ADAPT THINGS BACK FOR MULTIPLE EPSILON VALUES
disp("======== RESULTS ========");
disp("");
disp("Average computation time: "+string(sum(time(:))/N));
disp("Robust = "+string(sum(res(:)==1))+" out of " + string(N) + " videos")
disp("");
save("results/robustness_results_v1", "res", "time", "epsilon", "met");



% output = predict(netonnx, test);
% tot_incorrect = 0;
% [~, predictedLabels] = max(output, [], 2);
% 
% for i=1:1000
%     if predictedLabels(i)-1 ~= label(i)
%         tot_incorrect = tot_incorrect + 1;
%     end
% end

%% Helper function
function [IS, xRand] = NullFrameAttack(x, nR, frameNum)

    for frameNum=1:20
        nullFrame = zeros(32, 32);
        sample = x;
        sample(:, :, frameNum) = nullFrame;

        output = predict(netonnx, sample);
        
        
    end
end

