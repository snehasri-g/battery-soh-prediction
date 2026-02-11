clc; clear; close all;
rng(1);

if exist('B0005.mat','file')
    load('B0005.mat');
else
    error('Dataset not found');
end

data = B0005.cycle;

capacity = [];
for i = 1:length(data)
    if isfield(data(i), 'type') && strcmpi(data(i).type, 'discharge')
        if isfield(data(i).data, 'Capacity')
            capacity(end+1,1) = data(i).data.Capacity;
        end
    end
end

SOH = capacity / capacity(1);

figure; plot(SOH,'LineWidth',1.8);
grid on; xlabel('Cycle Number'); ylabel('SOH'); title('Battery Capacity vs Cycle Number');
saveas(gcf,'soh_curve.png');

earlyCycles = 1:40;
SOH_train = SOH(earlyCycles);

figure; plot(SOH_train,'LineWidth',1.8);
grid on; xlabel('Cycle Number'); ylabel('SOH'); title('Early Cycle SOH Used for LSTM Training');
saveas(gcf,'early_training.png');

cycles = (1:length(SOH_train))';
p = polyfit(cycles, SOH_train, 1);
baseline_pred = polyval(p, cycles);
rmse_baseline = sqrt(mean((baseline_pred - SOH_train).^2));
fprintf('Baseline RMSE: %.5f\n', rmse_baseline);

window = 5;
XTrain = {};
YTrain = [];

for i = 1:(length(SOH_train)-window)
    seq = SOH_train(i:i+window-1);
    targ = SOH_train(i+window);
    XTrain{end+1,1} = seq(:)';
    YTrain(end+1,1) = targ;
end

inputSize = 1;
outputSize = 1;
numHiddenUnits = 50;

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(outputSize)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'GradientThreshold',1, ...
    'Plots','training-progress', ...
    'Verbose',false);

netLSTM = trainNetwork(XTrain, YTrain, layers, options);

dischargeCycles = find(strcmp({B0005.cycle.type}, 'discharge'));
numCycles = length(dischargeCycles);

avgV = zeros(numCycles,1);
stdV = zeros(numCycles,1);

for n = 1:numCycles
    d = B0005.cycle(dischargeCycles(n)).data;
    fieldsV = fieldnames(d);
    Vfield = fieldsV{contains(lower(fieldsV), 'volt')};
    V = d.(Vfield);
    avgV(n) = mean(V);
    stdV(n) = std(V);
end

SOH_used = SOH(1:length(avgV));
inputs = [avgV stdV];
target = SOH_used;

anfis_data = [inputs target];
nTrn = floor(0.7 * length(SOH_used));

trnData = anfis_data(1:nTrn,:);
chkData = anfis_data(nTrn+1:end,:);

fis0 = genfis1(trnData, 3, 'gbellmf');
opt = anfisOptions('InitialFIS', fis0);
opt.EpochNumber = 300;
opt.ValidationData = chkData;

[fis, trnErr, stepSize, chkFIS, chkErr] = anfis(trnData, opt);

SOH_full = SOH_used;
LSTM_full = zeros(length(SOH_full),1);
LSTM_full(1:window) = SOH_full(1:window);

for i = window+1:length(SOH_full)
    seq = LSTM_full(i-window:i-1);
    seq = seq(:)';
    p = predict(netLSTM, {seq});
    if iscell(p), p = cell2mat(p); end
    LSTM_full(i) = p;
end

residual_full = SOH_full - LSTM_full;

X_AF = [];
res_AF = residual_full(window+1:end);

for i = 1:length(SOH_full)-window
    X_AF = [X_AF; SOH_full(i:i+window-1)'];
end

anfis_data_full = [X_AF res_AF];
fis2_init = genfis1(anfis_data_full, 3, 'gbellmf');
fis2 = anfis(anfis_data_full, fis2_init, 80);

residual_pred_full = evalfis(fis2, X_AF);

valStart = nTrn + 1;
valCycles = valStart:length(SOH_used);
y_actual = SOH_used(valStart:end);
y_pred_anfis = evalfis(fis, chkData(:,1:2));

figure;
plot(valCycles, y_actual,'LineWidth',1.6); hold on;
plot(valCycles, y_pred_anfis,'--','LineWidth',1.6);
grid on; xlabel('Cycle Number'); ylabel('SOH');
legend('Actual SOH','ANFIS Predicted SOH');
title('ANFIS Validation Performance (Actual vs Predicted)');
saveas(gcf,'anfis_prediction.png');

SOH_fused_full = LSTM_full;
SOH_fused_full(window+1:end) = LSTM_full(window+1:end) + residual_pred_full;

figure;
plot(SOH_full,'LineWidth',2); hold on;
plot(SOH_fused_full,'--','LineWidth',2);
xlabel('Cycle Number'); ylabel('SOH'); grid on;
title('FINAL FULL-LIFE HYBRID LSTM + ANFIS SOH PREDICTION');
saveas(gcf,'hybrid_prediction.png');

rmse_lstm = sqrt(mean((SOH_full - LSTM_full).^2));
rmse_hybrid = sqrt(mean((SOH_full - SOH_fused_full).^2));

fprintf('LSTM RMSE: %.5f\n', rmse_lstm);
fprintf('Hybrid RMSE: %.5f\n', rmse_hybrid);
