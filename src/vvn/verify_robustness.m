disp("Running robustness verification...");

if ~exist("./results", 'dir')
    mkdir("results");
end

% Load the model
models = ["C3D_bcss_zoom_in_model_op11_v1.onnx"];
netonnx = importONNXNetwork(models, "InputDataformats", "BCSS", "OutputDataFormats", "BC");
net = matlab2nnv(netonnx);
disp("finished loading model.");

% Load the dataset
data = readNPY('../datasets/numpy/mnistvideo_zoom_out_32x32_test_data_seq.npy');
labels = readNPY('../datasets/numpy/mnistvideo_zoom_out_32x32_test_labels_seq.npy');
data_squeezed = permute(squeeze(data), [3, 4, 2, 1]);
test = data_squeezed(:, :, :, :);

numClasses = 10;
n = 10; % Number of images to evaluate per class
N = n * numClasses; % Total number of samples to evaluate
nR = 297; % Number of videos to sample

% Define reachability options
reachOptions.reachMethod = 'relax-star-area';
reachOptions.relaxFactor = 0.5;

% Size of attack
epsilon = [1/255; 2/255; 3/255];
nE = length(epsilon);

% Results
res = zeros(N, nE);
time = zeros(N, nE);
met = repmat("relax", [N, nE]);


% Verification
for ms = 1:length(models)

    % Load the model
    modelName = models(ms);
    netonnx = importONNXNetwork("../", "InputDataFormats", "BCSS", "OutputDataFormats", "BC");
    net = matlab2nnv(netonnx);
    net.OutputSize = numClasses;
    disp("Running model: ");

    % Define reachability options
    reachOptions.reachMethod = 'relax-star-area';
    reachOptions.relaxFactor = 0.5;

    % Verify over each epsilon perturbation
    for e = 1:nE
        eps = epsilon(e);

        for i = 1:N
            [IS, xRand] = L_inf_attack(test(:,:,:,i), eps, nR, numClasses);
            t = tic;
            predictedLabels = predict(netonnx, xRand);
            [~, predictedLabels] = max(predictedLabels, [], 2);
            if any(predictedLabels-1 ~= labels(i))
                res(i, e) = 0;
                time(i, e) = toc(t);
                met(i, e) = "counterexample";
                continue;
            end

            try
                temp = net.verify_robustness(IS, reachOptions, labels(i)+1);
                if temp ~= 1 && temp ~= 0
                    reachOptions = struct;
                    reachOptions.reachMethod = 'approx-star';
                    temp = net.verify_robustness(IS, reachOptions, labels(i)+1);
                    met(i, e) = 'approx';
                end

            catch ME
                met(i, e) = ME.message;
                temp = -1;
            end

            res(i, e) = temp;
            time(i, e) = toc(t);

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

    end
end


% for i=1:N
%     fprintf("Iteration number : %d \n", i);
%     [IS, xRand] = L_inf_attack(test(:,:,:,i), epsilon, nR, 10);
%     t = tic;
%     predictedLabels = predict(netonnx, xRand);
%     [~, predictedLabels] = max(predictedLabels, [], 2);
%     if any(predictedLabels-1 ~= labels(i))
%         res(i) = 0;
%         time(i) = toc(t);
%         met(i) = "counterexample";
%         continue;
%     end
% 
%     try
%         temp = net.verify_robustness(IS, reachOptions, labels(i)+1);
%         if temp ~= 1 && temp ~= 0
%             reachOptions = struct;
%             reachOptions.reachMethod = 'approx-star';
%             temp = net.verify_robustness(IS, reachOptions, labels(i)+1);
%             met(i) = 'approx';
%         end
%     catch ME
%         met(i) = ME.message;
%         temp = -1;
%     end
% 
%     res(i) = temp;
%     time(i) = toc(t);
% 
%     reachOptions.reachMethod = 'relax-star-area';
%     reachOptions.relaxFactor = 0.5;
% end

%% Helper function
function [IS, xRand] = L_inf_attack(x, epsilon, nR, frameNum)
    lb = x;
    ub = x;

    lb(16, 16, frameNum) = x(16, 16, frameNum) - epsilon;
    if lb(16, 16, frameNum) < 0
        lb(16, 16, frameNum) = 0;
    end

    ub(16, 16, frameNum) = x(16, 16, frameNum) + epsilon;
    if ub(16, 16, frameNum) > 1
        ub(16, 16, frameNum) = 1;
    end

    IS = ImageStar(single(lb), single(ub));

    lb = reshape(lb, [20480, 1]);
    ub = reshape(ub, [20480, 1]);

    xB = Box(single(lb), single(ub));
    xRand = xB.sample(nR);
    xRand = reshape(xRand, [32, 32, 20, nR]);
    xRand(:, :, :, nR+1) = x;
    xRand(:, :, :, nR+2) = IS.im_lb;
    xRand(:, :, :, nR+3) = IS.im_ub;
end


