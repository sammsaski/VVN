% This script is modified to skip instances 62 + 87 because they took too
% long.

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

    % INDICES
    ind = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 91, 92, 95, 98, 100, 101, 103, 104, 105, 108, 129, 131, 133, 134, 135, 136, 137, 140, 143, 144, 145, 146, 147, 148, 149, 150, 152, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 169, 176, 272, 282, 285, 287, 288, 289, 290, 291, 292, 293, 295, 299, 300, 301, 304, 305, 309, 316, 320, 328, 330, 331, 332, 335, 336, 345, 368, 380, 395, 422, 573, 576, 579, 581, 600, 605, 607, 618, 628, 632, 822, 823, 825, 834, 838, 848, 863, 866, 870, 872];

    %%%%%%%%%%%%%%%%
    % VERIFICATION %
    %%%%%%%%%%%%%%%%
    for e=1:nE % we skip the first one because we completed that before
        fprintf('Starting verification with epsilon %d \n', epsilon(e));
        eps = epsilon(e);

        for index_num=1:100
            i = ind(index_num);
        
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
