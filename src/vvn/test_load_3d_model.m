% KEEP THIS FILE FOR SHOWING AN EXAMPLE OF FLATTEN3DINTO2DLAYER ERROR IN
% NNV

modelName = "C3D_zoom_out_3d_32x32.onnx";
netonnx = importONNXNetwork("../../models/" + modelName, "InputDataFormats", "TBCSS", "OutputDataFormats", "BC");
%netonnx = importONNXNetwork("../../models/" + modelName);
%analyzeNetwork(netonnx);
net = matlab2nnv(netonnx);
net.OutputSize = 10;
disp("Finished loading model: " + modelName);

% Load data
data = readNPY('../../data/3D/ZoomOut/mnistvideo_32x32_test_data_seq.npy');
labels = readNPY('../../data/3D/ZoomOut/mnistvideo_32x32_test_labels_seq.npy');

reshaped_data = permute(data, [1, 3, 2, 4, 5]); % to match BCSSS
data_squeezed = squeeze(reshaped_data);
datacopy = data_squeezed(:,:,:,:);

% Get a single sample + label, define epsilon
sample = datacopy(1,:,:,:); % get the first sample
sample = squeeze(sample);
label = labels(1);
epsilon = 1/255;

% Visualize a frame in the video
%figure;
%imshow(sample(1,:,:))

% Trying to get the network to work ; this is just testing code. can
% delete.
% sample = reshape(sample, [32, 32, 8, 1]);
% net.evaluate(sample)

%% Create the perturbed sample
% Perturb the sample on frame indexed at frame_num
p_frame_lower = sample(1,:,:) - epsilon; % was (:,:,1,:)
p_frame_upper = sample(1,:,:) + epsilon; % was (:,:,1,:)

% Create the lowerbound + upperbound
lb = sample;
ub = sample;

% Add the perturbation
lb(1,:,:) = p_frame_lower; % was (:,:,1,:)
lb = squeeze(lb);
ub(1,:,:) = p_frame_upper; % was (:,:,1,:)
ub = squeeze(ub);

% Create the volume star
lb_min = zeros(8, 32, 32);
ub_max = ones(8, 32, 32);
lb_clip = max(lb, lb_min);
ub_clip = min(ub, ub_max);

%% Create the Volume Star
VS = VolumeStar(lb_clip, ub_clip);

%% Verification
% Evaluate lower and upper bounds
LB_outputs = net.evaluate(lb_clip);
[~, LB_Pred] = max(LB_outputs);
UB_outputs = net.evaluate(ub_clip);
[~, UB_Pred] = max(UB_outputs);

Y_outputs = net.evaluate(sample);
[~, yPred] = max(Y_outputs);

% Now for verification
reachOptions = struct;
reachOptions.reachMethod = 'approx-star';

% Verification
t = tic;
res_approx = net.verify_robustness(VS, reachOptions, label);

if res_approx == 1
    disp("Neural network is verified to be robust!")
else
    disp("Unknown result")
end

fprintf("Res approx: %d", res_approx);

toc(t);

%% Let's visualize the ranges for every possible output
R = net.reachSet{end};

[lb_out, ub_out] = R.getRanges;
lb_out = squeeze(lb_out);
ub_out = squeeze(ub_out);

mid_range = (lb_out + ub_out) / 2;
range_size = ub_out - mid_range;

% Label for x-axis
x = [0 1 2 3 4 5 6 7 8 9];

figure;
errorbar(x, mid_range, range_size, '.');
hold on;
xlim([-0.5, 9.5]);
scatter(x, Y_outputs, 'x', 'MarkerEdgeColor', 'r');
scatter(x, LB_outputs, 'x', 'MarkerEdgeColor', 'b');
scatter(x, UB_outputs, 'x', 'MarkerEdgeColor', 'g');