%% Load the data + define static vars
% Load network
modelName = "C3D_small_bcss_zoom_out_model_op11_v6.onnx";
netonnx = importONNXNetwork("../../models/" + modelName, "InputDataFormats", "BCSS", "OutputDataFormats", "BC");

% Create NNV model
net = matlab2nnv(netonnx);
net.OutputSize = 10;

% Load data
data = readNPY('../../data/mnistvideo8frame_zoom_out_32x32_test_data_seq.npy');
labels = readNPY('../../data/mnistvideo8frame_zoom_out_32x32_test_labels_seq.npy');
data_squeezed = permute(squeeze(data), [3, 4, 2, 1]);
datacopy = data_squeezed(:, :, :, :);

% Get a single sample + label, define epsilon
sample = datacopy(:,:,:,1); % get the first sample
label = labels(1);
epsilon = 1/255;

disp("done loading sample")

% Visualize a frame in the video
%figure;
%imshow(sample(:,:,1))

% results = zeros(3, 8);

%% Create the perturbed sample
% Perturb the sample on frame indexed at frame_num
p_frame_lower = sample(:,:,1) - epsilon; % was (:,:,1)
p_frame_upper = sample(:,:,1) + epsilon; % was (:,:,1)

% Create the lowerbound + upperbound
lb = sample;
ub = sample;

% Add the perturbation
lb(:,:,1) = p_frame_lower; % was (:,:,1)
ub(:,:,1) = p_frame_upper; % was (:,:,1)

% Create the image star
lb_min = zeros(32, 32);
ub_max = ones(32, 32);
lb_clip = max(lb, lb_min);
ub_clip = min(ub, ub_max);

%% Create the Image Star
IS = ImageStar(lb_clip, ub_clip);

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
res_approx = net.verify_robustness(IS, reachOptions, label);

if res_approx == 1
    disp("Neural network is verified to be robust!")
else
    disp("Unknown result")
end

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