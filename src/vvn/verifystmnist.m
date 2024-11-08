function [res, time, met] = verifystmnist(smpLen, attackType, verAlg, index, epsIndex)
    %
    % smpLen (int)        : the length of a sample (video) in the dataset. either 4, 8, or 16.
    % attackType (string) : the type of video verification. either "single_frame" or "all_frames".
    % verAlg (string)     : the verification algorithm to use. either "relax" or "approx".
    % index (int)         : the index into the dataset to get the targeted sample to verify.
    % epsIndex (int)      : to help us select the epsilon value we would like to use for the attack.
    %

    if smpLen ~= 4 && smpLen ~= 8 && smpLen ~= 16 && smpLen ~= 32 && smpLen ~= 64
        printf("smpLen argument was invalid. Must be 4, 8, 16, 32, or 64.")
        return
    end

    if attackType ~= "single_frame" && attackType ~= "all_frames"
        printf("attackType argument was invalid. Must be 'single_frame' or 'all_frames'.")
        return
    end

    if verAlg ~= "relax" && verAlg ~= "approx"
        printf("verAlg argument was invalid. Must be 'relax' or 'approx'.")
        return
    end

    fprintf("Running robustness verification on STMNIST %df dataset...", smpLen);

    % Load data
    data = readNPY(sprintf("data/STMNIST/test/stmnistvideo_%df_test_data_seq.npy", smpLen));
    labels = readNPY(sprintf("data/STMNIST/test/stmnistvideo_%df_test_labels_seq.npy", smpLen));

    % Preprocessing
    reshaped_data = permute(data, [1, 3, 4, 5, 2]); % to match BCSSS
    datacopy = reshaped_data(:,:,:,:,:);

    % Experimental variables
    numClasses = 10;
    n = 10; % Number of images to evaluate per class
    N = n * numClasses; % Total number of samples to evaluate

    % Size of attack
    epsilon = [1/255; 2/255; 3/255];
    nE = length(epsilon);

    % Load the model
    modelName = sprintf("stmnist_%df.onnx", smpLen);
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
    end
    
    % Make predictions on test set to check that we are verifying correct
    % samples
    outputLabels = zeros(length(datacopy));

    s = datacopy(index,:,:,:,:);
    s = squeeze(s);

    output = net.evaluate(s);
    [~, P] = max(output);

    %%%%%%%%%%%%%%%%
    % VERIFICATION %
    %%%%%%%%%%%%%%%%

    eps = epsilon(epsIndex);
    fprintf('Starting verification with epsilon %d \n', eps);
    
    % Get the sample
    sample = squeeze(datacopy(index,:,:,:,:));
    
    % Perform L_inf attack
    VS = L_inf_attack(sample, eps, smpLen);
    t = tic;

    % NEED THIS HERE SO MET EXISTS
    met = verAlg;
 
    try
        % run verification algorithm
        temp = net.verify_robustness(VS, reachOptions, labels(index)+1);
                
    catch ME
        met = ME.message;
        temp = -1;
    end
    
    res = temp;
    time = toc(t);

    % % Get the reachable sets 
    % R = net.reachSet{end};
    % 
    % % Get ranges for each output index
    % [lb_out, ub_out] = R.getRanges;
    % lb_out = squeeze(lb_out);
    % ub_out = squeeze(ub_out);
    % 
    % % Get middle point for each output and range sizes
    % mid_range = (lb_out + ub_out)/2;
    % range_size = ub_out - mid_range;
    % 
    % % Label for x-axis
    % x = [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42];
    % 
    % % Visualize set ranges and evaluation points
    % disp(labels(index)+1);
    % disp(P);
    % disp(res);
    % %fig = figure('Visible', 'off');
    % figure;
    % errorbar(x, mid_range, range_size, '.');
    % hold on;
    % xlim([-0.5, 42.5]);
    % scatter(x, output, 'x', 'MarkerEdgeColor', 'r');

    % Save the figure
    % saveas(fig, sprintf("figs/range_plots/%s/%s/%s/%d/fig_%d_label=%d_eps=%d.png", dsVar, attackType, verAlg, smpLen, index, label, epsIndex)); 

    % Close the figure
    % close(fig);
end

%% Helper Functions
function VS = L_inf_attack(x, epsilon, numFrames)
    lb = squeeze(x);
    ub = squeeze(x);

    % Perturb the frames
    for fn=1:numFrames
        lb(fn, :, :, :) = x(fn, :, :, :) - epsilon;
        ub(fn, :, :, :) = x(fn, :, :, :) + epsilon;
    end

    % Reshape for conversion to VolumeStar
    % lb = permute(lb, [2 3 1 4]);
    % ub = permute(ub, [2 3 1 4]);

    % Clip the perturbed values to be between 0-1
    lb_min = zeros(numFrames, 10, 10, 2);
    ub_max = ones(numFrames, 10, 10, 2);
    lb_clip = max(lb, lb_min);
    ub_clip = min(ub, ub_max);

    % Create the volume star
    VS = VolumeStar(lb_clip, ub_clip);
end

