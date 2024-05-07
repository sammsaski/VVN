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

% Visualize a frame in the video
%figure;
%imshow(sample(:,:,1))

% Perturb the sample on frame 1
p_frame_lower = sample(:,:,1) - epsilon;
p_frame_upper = sample(:,:,1) + epsilon;

% Create the lowerbound + upperbound
lb = sample;
ub = sample;

% Add the perturbation
lb(:,:,1) = p_frame_lower;
ub(:,:,1) = p_frame_upper;

% Create the image star
lb_min = zeros(32, 32);
ub_max = 255*ones(32, 32);
lb_clip = max(lb, lb_min);
ub_clip = min(ub, ub_max);
IS = ImageStar(lb_clip, ub_clip);

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



%% Helper function
function [IS, xRand] = L_inf_attack(x, epsilon, nR)
    lb = x;
    ub = x;

    lb(:, :, :) = x(:, :, :) - epsilon;
    ub(:, :, :) = x(:, :, :) + epsilon;

    % In nnv/examples/Tutorial/NN/MNIST/verify.m, the pixel values are
    % clipped. Is this normal?

    IS = ImageStar(single(lb), single(ub));

    lb = reshape(lb, [8192, 1]);
    ub = reshape(ub, [8192, 1]);

    xB = Box(single(lb), single(ub));
    xRand = xB.sample(nR);
    xRand = reshape(xRand, [32, 32, 8, nR]);
    xRand(:, :, :, nR+1) = x;
    xRand(:, :, :, nR+2) = IS.im_lb;
    xRand(:, :, :, nR+3) = IS.im_ub;
end