%% ROBUSTNESS VERIFICATION ON ZOOM OUT DATASET
disp("Running robustness verification on Zoom Out dataset...");

if ~exist("./results/3DZoomOut/", 'dir')
    mkdir("results/3DZoomOut/");
end


% Load the model
models = ["C3D_zoom_out_3d_32x32.onnx",]; % "R3D_bcss_zoom_out_model_op11_v1.onnx"];

% Load the dataset
data = readNPY('../../data/mnistvideo8frame_zoom_out_32x32_test_data_seq.npy');
labels = readNPY('../../data/mnistvideo8frame_zoom_out_32x32_test_labels_seq.npy');
data_squeezed = permute(squeeze(data), [3, 4, 2, 1]);
datacopy = data_squeezed(:, :, :, :);

numClasses = 10;
n = 10; % Number of images to evaluate per class
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
    net.OutputSize = numClasses;
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
        
        % Track number of samples verified
        classIndex = zeros(numClasses, 1);

        i = 1;
        while any(classIndex ~= 10)

            % If we've already verified 10 samples of this class, then skip
            if classIndex(labels(i)+1) == 10
                i = i + 1;
                continue;
            end

            % If the current sample is already incorrectly classified, then
            % skip
            if outputLabels(i) ~= labels(i)+1
                i = i + 1;
                continue;
            end

            % Otherwise, add 1 to class index
            classIndex(labels(i)+1) = classIndex(labels(i)+1) + 1;
            
            iterationNum = sum(classIndex);
            fprintf('Iteration %d \n', iterationNum);

            [IS, xRand] = L_inf_attack(datacopy(:,:,:,i), eps, nR, 4); % choose frame number 4
            t = tic;
            predictedLabels = predict(netonnx, xRand);
            [~, predictedLabels] = max(predictedLabels, [], 2);

            if any(predictedLabels-1 ~= labels(i))
                res(iterationNum, e) = 0;
                time(iterationNum, e) = toc(t);
                met(iterationNum, e) = "counterexample";
                continue;
            end

            try
                temp = net.verify_robustness(IS, reachOptions, labels(i)+1);
                if temp ~= 1 && temp ~= 0
                    reachOptions = struct;
                    reachOptions.reachMethod = 'approx-star';
                    temp = net.verify_robustness(IS, reachOptions, labels(i)+1);
                    met(iterationNum, e) = 'approx';
                end

            catch ME
                met(iterationNum, e) = ME.message;
                temp = -1;
            end

            res(iterationNum, e) = temp;
            time(iterationNum, e) = toc(t);

            reachOptions.reachMethod = 'relax-star-area';
            reachOptions.relaxFactor = 0.5;

            % move to the next sample
            i = i + 1;
        end

        % Results MAKE SURE TO ADAPT THINGS BACK FOR MULTIPLE EPSILON VALUES
        disp("======== RESULTS ========");
        disp("");
        disp("Average computation time: "+string(sum(time(:,e))/N));
        disp("");
        disp("Robust = "+string(sum(res(:,e)==1))+" out of " + string(N) + " videos");
        disp("");
        save("results/ZoomOut/C3D_small_robustness_results_FPV", "res", "time", "epsilon", "met");

    end
end

%% ROBUSTNESS VERIFICATION ON ZOOM IN DATASET
disp("Running robustness verification on Zoom In dataset...");

if ~exist("./results/ZoomIn/", 'dir')
    mkdir("results/ZoomIn/");
end


% Load the model
models = ["C3D_small_bcss_zoom_in_model_op11_v6.onnx",]; % "R3D_bcss_zoom_in_model_op11_v1.onnx"];

% Load the dataset
data = readNPY('../../data/mnistvideo8frame_zoom_in_32x32_test_data_seq.npy');
labels = readNPY('../../data/mnistvideo8frame_zoom_in_32x32_test_labels_seq.npy');
data_squeezed = permute(squeeze(data), [3, 4, 2, 1]);
datacopy = data_squeezed(:, :, :, :);

numClasses = 10;
n = 10; % Number of images to evaluate per class
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
    net.OutputSize = numClasses;
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

        % Track number of samples verified
        classIndex = zeros(numClasses, 1);

        i = 1;
        while any(classIndex ~= 10)

            % If we've already verified 10 samples of this class, then skip
            if classIndex(labels(i)+1) == 10
                i = i + 1;
                continue;
            end

            % If the current sample is already incorrectly classified, then
            % skip
            if outputLabels(i) ~= labels(i)+1
                i = i + 1;
                continue;
            end
         
            % Otherwise, add 1 to class index
            classIndex(labels(i)+1) = classIndex(labels(i)+1) + 1;
            
            iterationNum = sum(classIndex);
            fprintf('Iteration %d \n', iterationNum);

            [IS, xRand] = L_inf_attack(datacopy(:,:,:,i), eps, nR, 4); % frame number 4 selected
            t = tic;
            predictedLabels = predict(netonnx, xRand);
            [~, predictedLabels] = max(predictedLabels, [], 2);
            if any(predictedLabels-1 ~= labels(i))
                res(iterationNum, e) = 0;
                time(iterationNum, e) = toc(t);
                met(iterationNum, e) = "counterexample";
                continue;
            end

            try
                temp = net.verify_robustness(IS, reachOptions, labels(i)+1);
                if temp ~= 1 && temp ~= 0
                    reachOptions = struct;
                    reachOptions.reachMethod = 'approx-star';
                    temp = net.verify_robustness(IS, reachOptions, labels(i)+1);
                    met(iterationNum, e) = 'approx';
                end

            catch ME
                met(iterationNum, e) = ME.message;
                temp = -1;
            end

            res(iterationNum, e) = temp;
            time(iterationNum, e) = toc(t);

            reachOptions.reachMethod = 'relax-star-area';
            reachOptions.relaxFactor = 0.5;
            
            % move to the next sample
            i = i + 1;
        end

        % Results MAKE SURE TO ADAPT THINGS BACK FOR MULTIPLE EPSILON VALUES
        disp("======== RESULTS ========");
        disp("");
        disp("Average computation time: "+string(sum(time(:,e))/N));
        disp("");
        disp("Robust = "+string(sum(res(:,e)==1))+" out of " + string(N) + " videos");
        disp("");
        save("results/ZoomIn/C3D_small_robustness_results_FPV", "res", "time", "epsilon", "met");

    end
end
