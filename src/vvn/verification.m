%% Zoom In - 4f
verify("zoom_in", 4, "all_frames");

%% Zoom Out - 4f
verify("zoom_out", 4, "all_frames");

%% Zoom In - 8f
verify("zoom_in", 8, "all_frames");

%% Zoom Out - 8f
verify("zoom_out", 8, "all_frames");

%% Zoom In - 16f
verify("zoom_in", 16, "all_frames");

%% Zoom Out - 16f
verify("zoom_out", 16, "all_frames");

%% Helper Functions
function [VS, xRand] = L_inf_attack(x, epsilon, nR, numFrames)
    lb = squeeze(x);
    ub = squeeze(x);

    % Perturb the frames
    for fn=1:numFrames
        lb(fn, :, :) = x(fn, :, :) - epsilon;
        ub(fn, :, :) = x(fn, :, :) + epsilon;
    end

    % Clip the perturbed values to be between 0-1
    lb_min = zeros(numFrames, 32, 32);
    ub_max = ones(numFrames, 32, 32);
    lb_clip = max(lb, lb_min);
    ub_clip = min(ub, ub_max);

    % Create the volume star
    VS = VolumeStar(lb_clip, ub_clip);

    % Create random images from initial set
    lb = reshape(lb, [numFrames*32*32, 1]);
    ub = reshape(ub, [numFrames*32*32, 1]);
    xB = Box(single(lb), single(ub));
    xRand = xB.sample(nR);
    xRand = reshape(xRand, [numFrames, 32, 32, nR]);
    xRand(:,:,:,nR+1) = x;
    xRand(:,:,:,nR+2) = VS.vol_lb;
    xRand(:,:,:,nR+3) = VS.vol_ub;
end

function [] = verify(dsVar, smpLen, vType)
    %
    % dsVar (string) : the dataset type. either "zoom_in" or "zoom_out".
    % smpLen (int): the length of a sample (video) in the dataset. either 4, 8, or 16.
    % vType (string) : the type of video verification. either "single_frame" or "all_frames".
    %

    % Check input arguments first.
    if dsVar ~= "zoom_in" && dsVar ~= "zoom_out"
        printf("dsVar argument was invalid. Must be 'zoom_in' or 'zoom_out'.")
        return
    end

    if smpLen ~= 4 && smpLen ~= 8 && smpLen ~= 16
        printf("smpLen argument was invalid. Must be 4, 8, or 16.")
        return
    end

    if vType ~= "single_frame" && vType ~= "all_frames"
        printf("vType argument was invalid. Must be 'single_frame' or 'all_frames'.")
        return
    end

    % Get alternative names used in file naming.
    dsVarCaps = "";
    dsVarShort = "";

    if dsVar == "zoom_in"
        dsVarCaps = "ZoomIn";
        dsVarShort = "zoomin";
    else
        dsVarCaps = "ZoomOut";
        dsVarShort = "zoomout";
    end
    

    fprintf("Running robustness verification on %s dataset...", dsVarCaps);

    if ~exist(sprintf("../../results/all_frames/%s/", dsVarCaps), "dir") % can "dir" be double quotes? it was apostrophes
        mkdir(sprintf("../../results/all_frames/%s", dsVarCaps));
    end

    % Load data
    data = readNPY(sprintf("../../data/%s/test/mnistvideo_%s_%df_test_data_seq.npy", dsVarCaps, dsVar, smpLen));
    labels = readNPY(sprintf("../../data/%s/test/mnistvideo_%s_test_labels_seq.npy", dsVarCaps, dsVar));

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
    modelName = sprintf("%s_%df.onnx", dsVarShort, smpLen);
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

    %%%%%%%%%%%%%%%%
    % VERIFICATION %
    %%%%%%%%%%%%%%%%
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
            [VS, xRand] = L_inf_attack(sample, eps, nR, smpLen);
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
        save(sprintf("../../results/%s/%s/%df", vType, dsVarCaps, smpLen), "res", "time", "epsilon", "met");
    end
end
