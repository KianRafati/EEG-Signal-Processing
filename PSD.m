addpath('C:\Program Files\eeglab2024.0');
% EEG1 = pop_biosig('wos1/chb01_02.edf');
EEG1 = pop_biosig('ws1/chb01_03.edf');

% Define parameters
epoch_length_minutes = 10; % Length of each epoch in minutes
epoch_length_samples = epoch_length_minutes * 60 * EEG1.srate; % Length of each epoch in samples

num_channels = size(EEG1.data, 1);
num_samples = size(EEG1.data, 2);
num_epochs = floor(num_samples / epoch_length_samples); % Number of complete epochs

% Initialize variables to store PSD and Shannon entropy results
window_length = 256; % Length of each segment for FFT
noverlap = 128; % Number of overlapping samples
nfft = 512; % Number of FFT points
fs = EEG1.srate; % Sampling frequency

psd_all_epochs = cell(num_channels, num_epochs);
shannon_entropy_all_epochs = zeros(num_channels, num_epochs);

% Split data into epochs and compute PSD and Shannon Entropy
for ep = 1:num_epochs
    % Define the sample range for the current epoch
    epoch_start = (ep-1) * epoch_length_samples + 1;
    epoch_end = ep * epoch_length_samples;
    
    for ch = 1:num_channels
        % Extract data for the current epoch and channel
        data = EEG1.data(ch, epoch_start:epoch_end);
        
        % Compute PSD using pwelch
        [pxx, f] = pwelch(data, window_length, noverlap, nfft, fs);
        
        % Normalize the PSD to get probability distribution
        pxx_norm = pxx / sum(pxx);
        
        % Compute Shannon Entropy
        shannon_entropy = -sum(pxx_norm .* log2(pxx_norm + eps)); % Adding eps to avoid log(0)
        
        % Store the results
        psd_all_epochs{ch, ep} = struct('pxx', pxx, 'f', f);
        shannon_entropy_all_epochs(ch, ep) = shannon_entropy;
    end
end

% Plot Shannon Entropy
figure;
for ch = 1:num_channels
    subplot(num_channels, 1, ch);
    plot(1:num_epochs, shannon_entropy_all_epochs(ch, :));
    title(['Shannon Entropy - Channel ', num2str(ch)]);
    if ch == num_channels
        xlabel('Epoch');
    end
    if ch == 1
        ylabel('Entropy');
    end
end

% Plot PSD for all epochs and all channels in one window
figure;
for ch = 1:num_channels
    for ep = 1:num_epochs
        subplot(num_channels, num_epochs, (ch-1)*num_epochs + ep);
        psd_struct = psd_all_epochs{ch, ep}; % Ensure psd{ch, ep} is a structure
        pxx_values = psd_struct.pxx; % Get the PSD values
        frequency_values = psd_struct.f; % Get the frequency values
        
        plot(frequency_values, 10*log10(pxx_values));
        if ch == 1
            title(['Epoch ', num2str(ep)]);
        end
        if ep == 1
            ylabel(['Ch ', num2str(ch)]);
        end
        if ch == num_channels
            xlabel('Frequency (Hz)');
        end
    end
end

% Add common labels for all subplots
han1 = axes('visible', 'off');
ylabel(han1, 'Power/Frequency (dB/Hz)');
xlabel(han1, 'Frequency (Hz)');


%=====================================================================
%Feature Selection
%=====================================================================


% Initialize arrays to store selected features
selected_features = struct;

% Hypothesized mean
mu0 = 0;

% Iterate over each channel
for ch = 1:num_channels
    % Collect feature vectors for each feature across all epochs
    pxx = []; shannon_entropy = []; mean_val = []; std_val = []; min_val = []; max_val = [];
   
    for ep = 1:num_epochs
        pxx = [pxx; features{ch, ep}.pxx];
        shannon_entropy = [shannon_entropy; features{ch, ep}.shannon_entropy];
        mean_val = [mean_val; features{ch, ep}.mean];
        std_val = [std_val; features{ch, ep}.std];
        min_val = [min_val; features{ch, ep}.min];
        max_val = [max_val; features{ch, ep}.max];
    end
   
    % Perform one-sample t-test for each feature
    [h_pxx, p_pxx] = ttest(pxx, mu0, 'Alpha', 0.001);
    [h_shannon, p_shannon] = ttest(shannon_entropy, mu0, 'Alpha', 0.001);
    [h_mean, p_mean] = ttest(mean_val, mu0, 'Alpha', 0.001);
    [h_std, p_std] = ttest(std_val, mu0, 'Alpha', 0.001);
    [h_min, p_min] = ttest(min_val, mu0, 'Alpha', 0.001);
    [h_max, p_max] = ttest(max_val, mu0, 'Alpha', 0.001);
   
    % Store selected features in the structure
    selected_features(ch).pxx = p_pxx < 0.001;
    selected_features(ch).shannon_entropy = p_shannon < 0.001;
    selected_features(ch).mean = p_mean < 0.001;
    selected_features(ch).std = p_std < 0.001;
    selected_features(ch).min = p_min < 0.001;
    selected_features(ch).max = p_max < 0.001;
end

% Display selected features for the first channel
disp('Selected features for channel 1:');
disp(selected_features(1));


%=====================================================================
% Training SVM
%=====================================================================

% Feature Extraction for Training Data
% A: 6 non-seizure epochs from a non-seizure data
A_EEG = pop_biosig('wos1/chb01_02.edf');
featuresA = [featureExtractor(A_EEG, 600), featureExtractor(A_EEG, 1200), featureExtractor(A_EEG, 1800), ...
             featureExtractor(A_EEG, 2400), featureExtractor(A_EEG, 3000), featureExtractor(A_EEG, 3600)];

% B: 2 non-seizure epochs from a seizure data
B_EEG = pop_biosig('ws1/chb01_04.edf');
featuresB = [featureExtractor(B_EEG, 600), featureExtractor(B_EEG, 3600)];

% C: 6 seizure epochs (seizure in last minute)
C_EEG1 = pop_biosig('ws1/chb01_03.edf');
featuresC = [featureExtractor(C_EEG1, 3056), featureExtractor(C_EEG1, 1527)];
C_EEG2 = pop_biosig('ws1/chb01_04.edf');
featuresC = [featuresC, featureExtractor(C_EEG2, 1527)];
C_EEG3 = pop_biosig('ws1/chb01_15.edf');
featuresC = [featuresC, featureExtractor(C_EEG3, 1792)];
C_EEG4 = pop_biosig('ws1/chb01_16.edf');
featuresC = [featuresC, featureExtractor(C_EEG4, 1075)];
C_EEG5 = pop_biosig('ws1/chb01_18.edf');
featuresC = [featuresC, featureExtractor(C_EEG5, 1780)];
C_EEG6 = pop_biosig('ws1/chb01_26.edf');
featuresC = [featuresC, featureExtractor(C_EEG6, 1922)];

num_features_per_epoch = 5;

% Define the number of epochs
num_epochs_per_block = 37; % Assumed number of epochs in each block

% Concatenate features for training
train_features = [featuresC(:, 1:num_epochs_per_block), ...
                  featuresC(:, num_epochs_per_block+1:2*num_epochs_per_block), ...
                  featuresC(:, 2*num_epochs_per_block+1:3*num_epochs_per_block), ...
                    featuresC(:, 3*num_epochs_per_block+1:4*num_epochs_per_block), ...
                    featuresA(:, 1:num_epochs_per_block), ...
                        featuresA(:, num_epochs_per_block+1:2*num_epochs_per_block), ...
                            featuresA(:, 2*num_epochs_per_block+1:3*num_epochs_per_block), ...
                                featuresA(:, 3*num_epochs_per_block+1:4*num_epochs_per_block), ...
                                    featuresA(:, 4*num_epochs_per_block+1:5*num_epochs_per_block), ...
                                        featuresB(:, 1:num_epochs_per_block)];


train_labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0];

% Concatenate features for testing
test_features = [featuresC(:, 4*num_epochs_per_block+1:5*num_epochs_per_block), ...
    featuresC(:, 5*num_epochs_per_block+1:6*num_epochs_per_block), ...
        featuresA(:, 5*num_epochs_per_block+1:6*num_epochs_per_block), ...
            featuresB(:, num_epochs_per_block+1:2*num_epochs_per_block)];

test_labels = [1, 1, 0, 0];

% Initialize matrices to store feature vectors
train_matrix = [];
test_matrix = [];

% Construct training matrix
for index = 1:10
    feature_vector = [];
    for ep = 1:num_epochs_per_block
        for ch = 1:23
            epoch_features = train_features{ch, (index-1)*num_epochs_per_block+ep};
            feature_vector = [feature_vector, ...
                              epoch_features.std, ...
                              epoch_features.min, ...
                              epoch_features.max, ...
                              epoch_features.shannon_entropy, ...
                              mean(epoch_features.pxx)];
        end
    end
    train_matrix = [train_matrix; feature_vector];
end

% Construct testing matrix
for index = 1:4
    feature_vector = [];
    for ep = 1:num_epochs_per_block
        for ch = 1:23
            epoch_features = test_features{ch, (index-1)*num_epochs_per_block+ep};
            feature_vector = [feature_vector, ...
                              epoch_features.std, ...
                              epoch_features.min, ...
                              epoch_features.max, ...
                              epoch_features.shannon_entropy, ...
                              mean(epoch_features.pxx)];
        end
    end
    test_matrix = [test_matrix; feature_vector];
end

% Train SVM
SVMModel = fitcsvm(train_matrix, train_labels, 'KernelFunction', 'linear', 'Standardize', true);
% Test SVM
predicted_labels = predict(SVMModel, test_matrix);

% Calculate confusion matrix
conf_matrix = confusionmat(test_labels, predicted_labels);
disp('Confusion Matrix:');
disp(conf_matrix);
% Plot confusion matrix
figure;
confusionchart(conf_matrix, {'Non-Seizure', 'Seizure'});
title('Confusion Matrix');
% Train and Test KNN Classifier

% Define number of neighbors (K) for KNN
K = 5;

% Train KNN
KNNModel = fitcknn(train_matrix, train_labels, 'NumNeighbors', K, 'Distance', 'minkowski');

% Test KNN
predicted_labels_knn = predict(KNNModel, test_matrix);
disp(predicted_labels);
disp(test_labels);
% Calculate confusion matrix for KNN
conf_matrix_knn = confusionmat(test_labels, predicted_labels_knn);
disp('KNN Confusion Matrix:');
disp(conf_matrix_knn);

% Plot confusion matrix for KNN
figure;
confusionchart(conf_matrix_knn, {'Non-Seizure', 'Seizure'});
title('KNN Confusion Matrix');

% Initialize variables for performance measures
num_samples = size(test_matrix, 1);
TP = zeros(num_samples, 1);
TN = zeros(num_samples, 1);
FP = zeros(num_samples, 1);
FN = zeros(num_samples, 1);

% Predict labels for test data using SVM and measure latency
tic; % Start timer
SVMModel = fitcsvm(train_matrix, train_labels); % Train SVM model
predicted_labels_svm = predict(SVMModel, test_matrix); % Predict labels for test data
svm_latency = toc; % Stop timer and calculate latency

% Calculate Sensitivity and Specificity
for i = 1:num_samples
    if predicted_labels_svm(i) == 1 && test_labels(i) == 1
        TP(i) = 1; % True positive
    elseif predicted_labels_svm(i) == 0 && test_labels(i) == 0
        TN(i) = 1; % True negative
    elseif predicted_labels_svm(i) == 1 && test_labels(i) == 0
        FP(i) = 1; % False positive
    elseif predicted_labels_svm(i) == 0 && test_labels(i) == 1
        FN(i) = 1; % False negative
    end
end

sensitivity_svm = sum(TP) / (sum(TP) + sum(FN));
specificity_svm = sum(TN) / (sum(TN) + sum(FP));

disp(['SVM Sensitivity (True Positive Rate): ', num2str(sensitivity_svm * 100), '%']);
disp(['SVM Specificity (True Negative Rate): ', num2str(specificity_svm * 100), '%']);
disp(['SVM Latency: ', num2str(svm_latency), ' seconds']);

% Repeat for KNN classifier

% Predict labels for test data using KNN and measure latency
tic; % Start timer
KNNModel = fitcknn(train_matrix, train_labels, 'NumNeighbors', K, 'Distance', 'minkowski'); % Train KNN model
predicted_labels_knn = predict(KNNModel, test_matrix); % Predict labels for test data
knn_latency = toc; % Stop timer and calculate latency

% Calculate Sensitivity and Specificity for KNN
for i = 1:num_samples
    if predicted_labels_knn(i) == 1 && test_labels(i) == 1
        TP(i) = 1; % True positive
    elseif predicted_labels_knn(i) == 0 && test_labels(i) == 0
        TN(i) = 1; % True negative
    elseif predicted_labels_knn(i) == 1 && test_labels(i) == 0
        FP(i) = 1; % False positive
    elseif predicted_labels_knn(i) == 0 && test_labels(i) == 1
        FN(i) = 1; % False negative
    end
end

sensitivity_knn = sum(TP) / (sum(TP) + sum(FN));
specificity_knn = sum(TN) / (sum(TN) + sum(FP));

disp(['KNN Sensitivity (True Positive Rate): ', num2str(sensitivity_knn * 100), '%']);
disp(['KNN Specificity (True Negative Rate): ', num2str(specificity_knn * 100), '%']);
disp(['KNN Latency: ', num2str(knn_latency), ' seconds']);


%=====================================================================
% Evaluation
%=====================================================================

% Initialize variables for performance measures
num_samples = size(test_matrix, 1);
TP = zeros(num_samples, 1);
TN = zeros(num_samples, 1);
FP = zeros(num_samples, 1);
FN = zeros(num_samples, 1);

% Predict labels for test data using SVM
SVMModel = fitcsvm(train_matrix, train_labels); % Train SVM model
predicted_labels_svm = predict(SVMModel, test_matrix); % Predict labels for test data

% Calculate Sensitivity and Specificity
for i = 1:num_samples
    if predicted_labels_svm(i) == 1 && test_labels(i) == 1
        TP(i) = 1; % True positive
    elseif predicted_labels_svm(i) == 0 && test_labels(i) == 0
        TN(i) = 1; % True negative
    elseif predicted_labels_svm(i) == 1 && test_labels(i) == 0
        FP(i) = 1; % False positive
    elseif predicted_labels_svm(i) == 0 && test_labels(i) == 1
        FN(i) = 1; % False negative
    end
end

sensitivity = sum(TP) / (sum(TP) + sum(FN));
specificity = sum(TN) / (sum(TN) + sum(FP));

disp(['Sensitivity (True Positive Rate): ', num2str(sensitivity * 100), '%']);
disp(['Specificity (True Negative Rate): ', num2str(specificity * 100), '%']);

% Assuming train_matrix and train_labels are defined

% Number of folds for cross-validation
k = 5;
total_labels = [1,1,1,1,0,0,0,0,0,0,1,1,0,0];
total_data = [train_matrix;test_matrix];
indices = crossvalind('Kfold', total_labels, k);
% Initialize variables to store the performance metrics
sensitivity = zeros(k, 1);
specificity = zeros(k, 1);

% Cross-validation loop
for i = 1:k
    test1 = (indices == i);
    train1 = ~test1;
    % Train the SVM Classifier on training data
    svmModel = fitcsvm(total_data(train1,:), total_labels(train1), 'KernelFunction', 'linear', ...
        'BoxConstraint', 1);
    % Test the SVM Classifier on testing data
    predictions = predict(svmModel, total_data(test1,:));
    disp(predictions);
    disp(total_labels(test1));
end
%=====================================================================
% Function definitions
%=====================================================================

function data_before_seizure = getDataBeforeTime(EEG, T)
    duration_before_seizure = 600; % duration in seconds
    start_time = T - duration_before_seizure;
    end_time = T;    
    start_sample = start_time * EEG.srate + 1;
    end_sample = end_time * EEG.srate;    
    data_before_seizure = EEG.data(:, start_sample:end_sample);
end

function features = getFeature(epoch_data, srate)
    [pxx, f] = pwelch(epoch_data, 256, 128, 512, srate);
    pxx_norm = pxx / sum(pxx);    
    shannon_entropy = -sum(pxx_norm .* log2(pxx_norm + eps));    
    mean_val = mean(epoch_data, 2);
    std_val = std(epoch_data, 0, 2);
    min_val = min(epoch_data, [], 2);
    max_val = max(epoch_data, [], 2);
    features = struct('pxx', pxx, 'f', f, 'shannon_entropy', shannon_entropy, ...
                      'mean', mean_val, 'std', std_val, 'min', min_val, 'max', max_val);
end

function matrix_data = getMatrix(data, window_size, step_size, srate)
    window_samples = window_size * srate;
    step_samples = step_size * srate;
    num_samples = size(data, 2);
    num_epochs = floor((num_samples - window_samples) / step_samples) + 1;
   
    matrix_data = cell(num_epochs, 1);
   
    for ep = 1:num_epochs
        start_sample = (ep - 1) * step_samples + 1;
        end_sample = start_sample + window_samples - 1;
        matrix_data{ep} = data(:, start_sample:end_sample);
    end
end

%=====================================================================
% Feature Extraction
%=====================================================================


function features = featureExtractor(EEG,T)    
    % Define the time of seizure beginning and get data before seizure
    data_before_seizure = getDataBeforeTime(EEG, T);

    % Define window size and step size
    window_size = 16;
    step_size = 16;

    % Get matrix data
    matrix_data = getMatrix(data_before_seizure, window_size, step_size, EEG.srate);

    num_epochs = length(matrix_data);
    num_channels = size(EEG.data, 1);
    features = cell(num_channels, num_epochs);

    % Define the feature names
    feature_names = {'mean', 'std', 'min', 'max', 'shannon_entropy', 'mean_pxx'};

    % Extract features
    for ep = 1:num_epochs
        for ch = 1:num_channels
            epoch_data = matrix_data{ep}(ch, :);
            features{ch, ep} = getFeature(epoch_data, EEG.srate);
        end
    end

    % Display features for the first channel and first epoch
    % disp('Features for channel 1, epoch 1:');
    % disp(features{1, 1});
end


function [mean_accuracy, accuracies] = k_fold_cross_validation(train_matrix, train_labels, k, model_type)
    num_samples = size(train_matrix, 1);
    indices = crossvalind('Kfold', num_samples, k); % Create indices for K-fold CV
    
    accuracies = zeros(k, 1);
    
    for i = 1:k
        test_indices = (indices == i);
        train_indices = ~test_indices;
        
        if strcmp(model_type, 'SVM')
            SVMModel = fitcsvm(train_matrix(train_indices, :), train_labels(train_indices));
            predicted_labels = predict(SVMModel, train_matrix(test_indices, :));
        elseif strcmp(model_type, 'KNN')
            KNNModel = fitcknn(train_matrix(train_indices, :), train_labels(train_indices));
            predicted_labels = predict(KNNModel, train_matrix(test_indices, :));
        else
            error('Unsupported model type.');
        end
        
        % Evaluate accuracy
        accuracies(i) = sum(predicted_labels == train_labels(test_indices)) / sum(test_indices);
    end
    
    mean_accuracy = mean(accuracies);
end

function mean_accuracy = leave_one_out_cross_validation(train_matrix, train_labels, model_type)
    num_samples = size(train_matrix, 1);
    accuracies = zeros(num_samples, 1);
    
    for i = 1:num_samples
        test_indices = i;
        train_indices = setdiff(1:num_samples, test_indices);
        
        if strcmp(model_type, 'SVM')
            SVMModel = fitcsvm(train_matrix(train_indices, :), train_labels(train_indices));
            predicted_label = predict(SVMModel, train_matrix(test_indices, :));
        elseif strcmp(model_type, 'KNN')
            KNNModel = fitcknn(train_matrix(train_indices, :), train_labels(train_indices));
            predicted_label = predict(KNNModel, train_matrix(test_indices, :));
        else
            error('Unsupported model type.');
        end
        
        % Evaluate accuracy
        accuracies(i) = predicted_label == train_labels(test_indices);
    end
    
    mean_accuracy = mean(accuracies);
end
