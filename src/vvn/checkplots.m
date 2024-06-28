function [res, label] = checkplots(dsVar, smpLen, attackType, verAlg, index, epsIndex)
    %
    % dsVar (string)      : the dataset type. either "zoom_in" or "zoom_out".
    % smpLen (int)        : the length of a sample (video) in the dataset. either 4, 8, or 16.
    % attackType (string) : the type of video verification. either "single_frame" or "all_frames".
    % verAlg (string)     : the verification algorithm to use. either "relax" or "approx".
    % index (int)         : the index into the dataset to get the targeted sample to verify.
    % epsIndex (int)      : to help us select the epsilon value we would like to use for the attack.
    % outputFile (string) : where to save the plot
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

    if verAlg ~= "relax" && verAlg ~= "approx" && verAlg ~= "exact"
        printf("verAlg argument was invalid. Must be 'relax', 'approx', or 'exact'.")
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
    % data = readNPY(sprintf("../../data/%s/test/mnistvideo_%s_%df_test_data_seq.npy", dsVarCaps, dsVar, smpLen));
    data = readNPY(sprintf("data/%s/test/mnistvideo_%s_%df_test_data_seq.npy", dsVarCaps, dsVar, smpLen));
    % labels = readNPY(sprintf("../../data/%s/test/mnistvideo_%s_test_labels_seq.npy", dsVarCaps, dsVar));
    labels = readNPY(sprintf("data/%s/test/mnistvideo_%s_test_labels_seq.npy", dsVarCaps, dsVar));

    % Preprocessing
    reshaped_data = permute(data, [1, 3, 2, 4, 5]); % to match BCSSS
    data_squeezed = squeeze(reshaped_data);
    datacopy = data_squeezed(:,:,:,:);

    % Experimental variables
    numClasses = 10;
    n = 10; % Number of images to evaluate per class
    N = n * numClasses; % Total number of samples to evaluate

    % Size of attack
    epsilon = [1/255; 2/255; 3/255];
    nE = length(epsilon);

    % Load the model
    modelName = sprintf("%s_%df.onnx", dsVarShort, smpLen);
    % netonnx = importONNXNetwork("../../models/" + modelName, "InputDataFormats", "TBCSS", "OutputDataFormats", "BC");
    netonnx = importONNXNetwork("models/" + modelName, "InputDataFormats", "TBCSS", "OutputDataFormats", "BC");
    net = matlab2nnv(netonnx);
    net.OutputSize = numClasses;
    disp("Finished loading model: " + modelName);

    % Verification settings
    reachOptions = struct;
    if verAlg == "relax"
        reachOptions.reachMethod = "relax-star-area";
        reachOptions.relaxFactor = 0.5;
    elseif verAlg == "approx"
        reachOptions.reachMethod = "approx-star";
    else
        reachOptions.reachMethod = "exact-star"; % TODO: fix this
    end

    %%%%%%%%%%%%%%%%
    % VERIFICATION %
    %%%%%%%%%%%%%%%%

    eps = epsilon(epsIndex);
    fprintf('Starting verification with epsilon %d \n', eps);
    
    % Get the sample + label
    sample = squeeze(datacopy(index,:,:,:));
    label = labels(index);

    % Evaluate the input video
    Y_outputs = net.evaluate(sample);
    [~, yPred] = max(Y_outputs);
    
    % Perform L_inf attack
    VS = L_inf_attack(sample, eps, smpLen);
    t = tic;

    % Run verification 
    res = net.verify_robustness(VS, reachOptions, label+1);
    
    % Stop the clock
    toc(t);

    % Get the reachable sets 
    R = net.reachSet{end};

    % Get ranges for each output index
    [lb_out, ub_out] = R.getRanges;
    lb_out = squeeze(lb_out);
    ub_out = squeeze(ub_out);

    % Get middle point for each output and range sizes
    mid_range = (lb_out + ub_out)/2;
    range_size = ub_out - mid_range;

    % Label for x-axis
    x = [0 1 2 3 4 5 6 7 8 9];

    % Visualize set ranges and evaluation points
    fig = figure('Visible', 'off');
    errorbar(x, mid_range, range_size, '.');
    hold on;
    xlim([-0.5, 9.5]);
    scatter(x, Y_outputs, 'x', 'MarkerEdgeColor', 'r');

    % Save the figure
    saveas(fig, sprintf("figs/range_plots/%s/%s/%s/%d/fig_%d_label=%d_eps=%d.png", dsVar, attackType, verAlg, smpLen, index, label, epsIndex)); 

    % Close the figure
    close(fig);
end

%% Helper Functions
function VS = L_inf_attack(x, epsilon, numFrames)
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
end
