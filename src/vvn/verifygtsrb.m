function [res, time, met] = verifygtsrb(smpLen, attackType, verAlg, index, epsIndex)
    %
    % smpLen (int)        : the length of a sample (video) in the dataset. either 4, 8, or 16.
    % attackType (string) : the type of video verification. either "single_frame" or "all_frames".
    % verAlg (string)     : the verification algorithm to use. either "relax" or "approx".
    % index (int)         : the index into the dataset to get the targeted sample to verify.
    % epsIndex (int)      : to help us select the epsilon value we would like to use for the attack.
    %

    if smpLen ~= 4 && smpLen ~= 8 && smpLen ~= 16
        printf("smpLen argument was invalid. Must be 4, 8, or 16.")
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

    fprintf("Running robustness verification on GTSRB %df dataset...", smpLen);

    % Load data
    data = readNPY(sprintf("data/GTSRB/test/gtsrbvideo_%df_test_data_seq.npy", smpLen));
    labels = readNPY("data/GTSRB/test/gtsrbvideo_test_labels_seq.npy");

    % Preprocessing
    reshaped_data = permute(data, [1, 2, 4, 5, 3]); % to match BCSSS
    datacopy = reshaped_data(:,:,:,:,:);

    % Experimental variables
    numClasses = 43;
    n = 5; % Number of images to evaluate per class
    N = n * numClasses; % Total number of samples to evaluate

    % Size of attack
    epsilon = [1/255; 2/255; 3/255];
    nE = length(epsilon);

    % Load the model
    modelName = sprintf("gtsrb_%df.onnx", smpLen);
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
    lb = permute(lb, [2 3 1 4]);
    ub = permute(ub, [2 3 1 4]);

    % Clip the perturbed values to be between 0-1
    lb_min = zeros(32, 32, numFrames, 3);
    ub_max = ones(32, 32, numFrames, 3);
    lb_clip = max(lb, lb_min);
    ub_clip = min(ub, ub_max);

    % Create the volume star
    VS = VolumeStar(lb_clip, ub_clip);
end

