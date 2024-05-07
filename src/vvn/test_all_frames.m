%% SCALABILITY TEST: ZOOM OUT
disp("Running robustness verification on Zoom Out dataset...");

if ~exist("./test_all_frames/ZoomOut/", 'dir')
    mkdir("test_all_frames/ZoomOut/");
end


% Load the model
models = ["C3D_small_bcss_zoom_out_model_op11_v6.onnx",]; % "R3D_bcss_zoom_out_model_op11_v1.onnx"];

% Load the dataset
data = readNPY('../../data/mnistvideo8frame_zoom_out_32x32_test_data_seq.npy');
labels = readNPY('../../data/mnistvideo8frame_zoom_out_32x32_test_labels_seq.npy');
data_squeezed = permute(squeeze(data), [3, 4, 2, 1]);
datacopy = data_squeezed(:, :, :, :);

numClasses = 1;
n = 1; % Number of images to evaluate per class
N = n * numClasses; % Total number of samples to evaluate
nR = 17; % Number of videos to sample

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
    netonnx = importONNXNetwork("../../models/" + modelName, "InputDataFormats", "BCSS", "OutputDataFormats", "BC");
    net = matlab2nnv(netonnx);
    net.OutputSize = 10;
    disp("Finished loading model: " + modelName);

    % Define reachability options
    reachOptions.reachMethod = 'relax-star-area';
    reachOptions.relaxFactor = 0.5; 

    % Make predictions on test set to check that we are verifying correct
    % samples
    output = predict(netonnx, datacopy);
    [~, outputLabels] = max(output, [], 2);

    % Verify over each epsilon perturbation
    for e = 1:nE
        fprintf('Starting verification with epsilon %d \n', epsilon(e));
        eps = epsilon(e);

        [IS, xRand] = L_inf_attack(datacopy(:,:,:,1), eps, nR);
        t = tic;
        predictedLabels = predict(netonnx, xRand);
        [~, predictedLabels] = max(predictedLabels, [], 2);

        if any(predictedLabels-1 ~= labels(1))
            res(1, e) = 0;
            time(1, e) = toc(t);
            met(1, e) = "counterexample";
            continue
        end

        try
            temp = net.verify_robustness(IS, reachOptions, labels(1)+1);
            if temp ~= 1 && temp ~= 0
                reachOptions = struct;
                reachOptions.reachMethod = 'approx-star';
                temp = net.verify_robustness(IS, reachOptions, labels(1)+1);
                met(1, e) = 'approx';
            end

        catch ME
            met(iterationNum, e, frameNum) = ME.message;
            temp = -1;
        end
    
        res(1, e) = temp;
        time(1, e) = toc(t);

        reachOptions.reachMethod = 'relax-star-area';
        reachOptions.relaxFactor = 0.5;

        % Results MAKE SURE TO ADAPT THINGS BACK FOR MULTIPLE EPSILON VALUES
        disp("======== RESULTS ========");
        disp("");
        disp("Average computation time: "+string(sum(time(:,e))));
        disp("");
        disp("Robust = "+string(sum(res(:,e)==1))+" out of " + string(N) + " videos");
        disp("");
        save("results/test_all_frames/ZoomOut/C3D_small_robustness_results_FPV", "res", "time", "epsilon", "met");
    end
end


%% Helper function
function [IS, xRand] = L_inf_attack(x, epsilon, nR)
    lb = x;
    ub = x;

    lb(:, :, :) = x(:, :, :) - epsilon;
    ub(:, :, :) = x(:, :, :) + epsilon;

    IS = ImageStar(single(lb), single(ub));

    lb = reshape(lb, [8192, 1]);
    ub = reshape(ub, [8192, 1]);

    xB = Box(single(lb), single(ub));
    xRand = xB.sample(nR);
    xRand = reshape(xRand, [32, 32, 8, nR]);
    xRand(:, :, :, nR+1) = x;
    xRand(:, :, :, nR+2) = IS.im_lb;
    xRand(:, :, :, nR+3) = IS.im_ub;
end