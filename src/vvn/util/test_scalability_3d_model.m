modelName = "C3D_zoom_out_3d_32x32.onnx";
netonnx = importONNXNetwork("../../models/" + modelName, "InputDataFormats", "TBCSS", "OutputDataFormats", "BC");
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
label = labels(1) + 1; % have to change label bc 1-indexing in MATLAB vs 0-indexing in Python
epsilon = 1/255;
numFrames = 1; % change this to perturb more frames

%% Create the perturbed sample
% Create the lowerbound + upperbound
lb = sample;
ub = sample;

for fn=1:numFrames
    % Add the perturbed frames
    lb(fn, :, :) = sample(fn, :, :) - epsilon;
    ub(fn, :, :) = sample(fn, :, :) + epsilon;
end

lb = squeeze(lb);
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
reachOptions.reachMethod = 'relax-star-area';
reachOptions.relaxFactor = 0.5;

% Verification
t = tic;
res_approx = net.verify_robustness(VS, reachOptions, label);

if res_approx == 1
    disp("Neural network is verified to be robust!")
else
    disp("Unknown result")
end

fprintf("Res approx: %d \n", res_approx);

toc(t);

% %% Let's visualize the ranges for every possible output
% R = net.reachSet{end};
% 
% [lb_out, ub_out] = R.getRanges;
% lb_out = squeeze(lb_out);
% ub_out = squeeze(ub_out);
% 
% mid_range = (lb_out + ub_out) / 2;
% range_size = ub_out - mid_range;
% 
% % Label for x-axis
% x = [0 1 2 3 4 5 6 7 8 9];
% 
% figure;
% errorbar(x, mid_range, range_size, '.');
% hold on;
% xlim([-0.5, 9.5]);
% scatter(x, Y_outputs, 'x', 'MarkerEdgeColor', 'r');
% scatter(x, LB_outputs, 'x', 'MarkerEdgeColor', 'b');
% scatter(x, UB_outputs, 'x', 'MarkerEdgeColor', 'g');