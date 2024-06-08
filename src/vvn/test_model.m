%% ROBUSTNESS VERIFICATION ON ZOOM OUT DATASET
disp("Running robustness verification on Zoom Out dataset...");

% Load data
data = readNPY('../../data/3D/ZoomOut/mnistvideo_32x32_test_data_seq.npy');
labels = readNPY('../../data/3D/ZoomOut/mnistvideo_32x32_test_labels_seq.npy');

% Preprocessing
reshaped_data = permute(data, [1, 3, 2, 4, 5]); % to match BCSSS
data_squeezed = squeeze(reshaped_data);
datacopy = data_squeezed(:,:,:,:);

% Experimental variables
numClasses = 10;
n = 10; % Number of images to evaluate per class
N = n * numClasses; % Total number of samples to evaluate
nR = 17; % Number of videos to sample

% Size of attack
epsilon = [1/255; 2/255; 3/255];
nE = length(epsilon);

% Results
res = zeros(N, nE);
time = zeros(N, nE);
met = repmat("relax", [N, nE]);

% Load the model
modelName = "C3D_zoom_out_3d_32x32.onnx";
netonnx = importONNXNetwork("../../models/" + modelName, "InputDataFormats", "TBCSS", "OutputDataFormats", "BC");
net = matlab2nnv(netonnx);
net.OutputSize = numClasses;
disp("Finished loading model: " + modelName);

% Make predictions on test set to check that we are verifying correct
% samples
outputLabels = zeros(length(datacopy), 1);

for i=1:length(datacopy)
    s = datacopy(i,:,:,:);
    s = squeeze(s);
    l = labels(i)+1; % convert from 0 to 1 indexing

    outputs = net.evaluate(s);
    [~, P] = max(outputs);

    % Add the prediction to the set of outputs on all samples
    outputLabels(i) = P;
end

disp("Finished getting predictions.");

%% Just for checking that the outputLabels are accurate
newLabels = labels + 1;

tc = 0;

for i=1:length(outputLabels)
    if newLabels(i) == outputLabels(i)
        tc = tc + 1;
    end
end

disp(tc);

%% Debugging -- This all looks okay?
classIndex = zeros(numClasses, 1);

i = 1;
while any(classIndex ~= n)

    if classIndex(labels(i)+1) == n % The indexing into labels could be wrong, try with newLabels to see if that fixes it
        i = i + 1;
        continue;
    end

    if outputLabels(i) ~= labels(i) + 1
        i = i + 1;
        continue;
    end
    
    % Otherwise, add 1 to class index
    classIndex(labels(i)+1) = classIndex(labels(i)+1) + 1;
    
    iterationNum = sum(classIndex);
    fprintf('Iteration %d \n', iterationNum);

    % Get the sample
    sample = squeeze(datacopy(i,:,:,:));
    
    % Perform L_inf attack
    [VS, xRand] = L_inf_attack(sample, eps, nR, 8);
    t = tic;

    % Falsification
    predictedLabels = zeros(nR+3, 1);
    for j=1:length(predictedLabels)
        s = xRand(:,:,:,j);
        s = squeeze(s);
    
        outputs = net.evaluate(s);
        [~, P] = max(outputs);
    
        % Add the prediction to the set of outputs on xRand
        predictedLabels(j) = P;
    end

    if any(predictedLabels ~= labels(i)+1)
        res(iterationNum, e) = 0;
        time(iterationNum, e) = toc(t);
        met(iterationNum, e) = "counterexample";
        continue;
    end

    i = i + 1;
end



%% Verification

for e=1:nE
    fprintf('Starting verification with epsilon %d \n', epsilon(e));
    eps = epsilon(e);
    
    % Track number of samples verified
    classIndex = zeros(numClasses, 1);

    i = 1;
    while any(classIndex ~= n)

        % If we've already verified 10 samples of this class, then skip
        if classIndex(labels(i)+1) == n
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

        % Get the sample
        sample = squeeze(datacopy(i,:,:,:));
        
        % Perform L_inf attack
        [VS, xRand] = L_inf_attack(sample, eps, nR, 8);
        t = tic;

        % Falsification
        predictedLabels = zeros(nR+3, 1);
        for j=1:length(predictedLabels)
            s = xRand(:,:,:,j);
            s = squeeze(s);
        
            outputs = net.evaluate(s);
            [~, P] = max(outputs);
        
            % Add the prediction to the set of outputs on xRand
            predictedLabels(j) = P;
        end

        if any(predictedLabels ~= labels(i)+1)
            res(iterationNum, e) = 0;
            time(iterationNum, e) = toc(t);
            met(iterationNum, e) = "counterexample";
            continue;
        end

        try
            temp = net.verify_robustness(VS, reachOptions, labels(i)+1);
            if temp ~= 1 && temp ~= 0
                reachOptions = struct;
                reachOptions.reachMethod = 'approx-star';
                temp = net.verify_robustness(VS, reachOptions, labels(i)+1);
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
    save("results/3D/ZoomOut/C3D", "res", "time", "epsilon", "met");
end


%% ROBUSTNESS VERIFICATION ON ZOOM IN DATASET
disp("Running robustness verification on Zoom In dataset...");

if ~exist("./results/3D/ZoomIn/", 'dir')
    mkdir("results/3D/ZoomIn/");
end

% Load data
data = readNPY('../../data/3D/ZoomIn/mnistvideo_32x32_test_data_seq.npy');
labels = readNPY('../../data/3D/ZoomIn/mnistvideo_32x32_test_labels_seq.npy');

% Preprocessing
reshaped_data = permute(data, [1, 3, 2, 4, 5]); % to match BCSSS
data_squeezed = squeeze(reshaped_data);
datacopy = data_squeezed(:,:,:,:);

% Experimental variables
numClasses = 10;
n = 10; % Number of images to evaluate per class
N = n * numClasses; % Total number of samples to evaluate
nR = 17; % Number of videos to sample

% Size of attack
epsilon = [1/255; 2/255; 3/255];
nE = length(epsilon);

% Results
res = zeros(N, nE);
time = zeros(N, nE);
met = repmat("relax", [N, nE]);

% Load the model
modelName = "C3D_zoom_in_3d_32x32.onnx";
netonnx = importONNXNetwork("../../models/" + modelName, "InputDataFormats", "TBCSS", "OutputDataFormats", "BC");
net = matlab2nnv(netonnx);
net.OutputSize = numClasses;
disp("Finished loading model: " + modelName);

% Make predictions on test set to check that we are verifying correct
% samples
outputLabels = zeros(length(datacopy));

for i=1:length(datacopy)
    s = datacopy(i,:,:,:);
    s = squeeze(s);
    l = labels(i)+1; % convert from 0 to 1 indexing

    outputs = net.evaluate(s);
    [~, P] = max(outputs);

    % Add the prediction to the set of outputs on all samples
    outputLabels(i) = P;
end

%% Verification

for e=1:nE
    fprintf('Starting verification with epsilon %d \n', epsilon(e));
    eps = epsilon(e);
    
    % Track number of samples verified
    classIndex = zeros(numClasses, 1);

    i = 1;
    while any(classIndex ~= n)

        % If we've already verified 10 samples of this class, then skip
        if classIndex(labels(i)+1) == n
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

        % Get the sample
        sample = squeeze(datacopy(i,:,:,:));
        
        % Perform L_inf attack
        [VS, xRand] = L_inf_attack(sample, eps, nR, 8);
        t = tic;

        % Falsification
        predictedLabels = zeros(nR+3, 1);
        for j=1:length(predictedLabels)
            s = xRand(:,:,:,j);
            s = squeeze(s);
        
            outputs = net.evaluate(s);
            [~, P] = max(outputs);
        
            % Add the prediction to the set of outputs on xRand
            predictedLabels(j) = P;
        end

        if any(predictedLabels ~= labels(i)+1)
            res(iterationNum, e) = 0;
            time(iterationNum, e) = toc(t);
            met(iterationNum, e) = "counterexample";
            continue;
        end

        try
            temp = net.verify_robustness(VS, reachOptions, labels(i)+1);
            if temp ~= 1 && temp ~= 0
                reachOptions = struct;
                reachOptions.reachMethod = 'approx-star';
                temp = net.verify_robustness(VS, reachOptions, labels(i)+1);
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
    save("results/3D/ZoomIn/C3D", "res", "time", "epsilon", "met");
end


%% Helper Function
function [VS, xRand] = L_inf_attack(x, epsilon, nR, numFrames)
    lb = squeeze(x);
    ub = squeeze(x);

    % Perturb the frames
    for fn=1:numFrames
        lb(fn, :, :) = x(fn, :, :) - epsilon;
        ub(fn, :, :) = x(fn, :, :) + epsilon;
    end

    % Clip the perturbed values to be between 0-1
    lb_min = zeros(8, 32, 32);
    ub_max = ones(8, 32, 32);
    lb_clip = max(lb, lb_min);
    ub_clip = min(ub, ub_max);

    % Create the volume star
    VS = VolumeStar(lb_clip, ub_clip);

    % Create random images from initial set
    lb = reshape(lb, [8192, 1]);
    ub = reshape(ub, [8192, 1]);
    xB = Box(single(lb), single(ub));
    xRand = xB.sample(nR);
    xRand = reshape(xRand, [8, 32, 32, nR]);
    xRand(:,:,:,nR+1) = x;
    xRand(:,:,:,nR+2) = VS.vol_lb;
    xRand(:,:,:,nR+3) = VS.vol_ub;
end
