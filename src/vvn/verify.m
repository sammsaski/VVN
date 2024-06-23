function [res, time, met] = verify(dsVar, smpLen, attackType, index, epsIndex)
    %
    % dsVar (string)      : the dataset type. either "zoom_in" or "zoom_out".
    % smpLen (int)        : the length of a sample (video) in the dataset. either 4, 8, or 16.
    % attackType (string) : the type of video verification. either "single_frame" or "all_frames".
    % index (int)         : the index into the dataset to get the targeted sample to verify.
    % epsIndex (int)      : to help us select the epsilon value we would like to use for the attack.
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

    if attackType ~= "single_frame" && attackType ~= "all_frames"
        printf("attackType argument was invalid. Must be 'single_frame' or 'all_frames'.")
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

    % Load the model
    modelName = sprintf("%s_%df.onnx", dsVarShort, smpLen);
    netonnx = importONNXNetwork("../../models/" + modelName, "InputDataFormats", "TBCSS", "OutputDataFormats", "BC");
    net = matlab2nnv(netonnx);
    net.OutputSize = numClasses;
    disp("Finished loading model: " + modelName);

    % Verification settings
    reachOptions = struct;
    reachOptions.reachMethod = 'relax-star-area';
    reachOptions.relaxFactor = 0.5;

    % Make predictions on test set to check that we are verifying correct
    % samples
    outputLabels = zeros(length(datacopy));

    s = datacopy(index,:,:,:);
    s = squeeze(s);
    l = labels(index) + 1;

    output = net.evaluate(s);
    [~, P] = max(output);

    %%%%%%%%%%%%%%%%
    % VERIFICATION %
    %%%%%%%%%%%%%%%%

    eps = epsilon(epsIndex);
    fprintf('Starting verification with epsilon %d \n', eps);
    
    % Get the sample
    sample = squeeze(datacopy(index,:,:,:));
    
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

    % NEED THIS HERE SO MET EXISTS
    met = "relax";

    if any(predictedLabels ~= labels(index)+1)
        res = 0;
        time = toc(t);
        met = "counterexample";
    else
        try
            temp = net.verify_robustness(VS, reachOptions, labels(index)+1);
            if temp ~= 1 && temp ~= 0
                reachOptions = struct;
                reachOptions.reachMethod = 'approx-star';
                temp = net.verify_robustness(VS, reachOptions, labels(index)+1);
                met = 'approx';
            end
    
        catch ME
            met = ME.message;
            temp = -1;
        end
    
        res = temp;
        time = toc(t);
    end
end

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
