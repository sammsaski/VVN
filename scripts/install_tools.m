%
% Setup NNV
%
cd /home/user/nnv/code/nnv
install;

%
% Setup npy-matlab
%
cd /home/user/npy-matlab
addpath('/home/user/npy-matlab/npy-matlab');
savepath;

fprintf('\nFinished installing NNV + npy-matlab.\n');

%
% Running subset of experiments
%
% disp("\nRunning subset of experiments.\n");
% cd /home/user/verify_malware/verify/nnv_verify
% run_subset;