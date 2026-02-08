tab = readtable('led3.csv', 'VariableNamingRule', 'preserve');

tab2 = tab;

tab.Country = [];

tab.Year = tab.Year - 2000;

uniqueStatus = unique(tab.Status);
for nr = 1:length(uniqueStatus)
    for i = 1:height(tab)
        if isequal(tab.Status{i}, uniqueStatus{nr})
            tab.Status{i} = nr-1;
        end
    end
end
tab.Status = cell2mat(tab.Status);

misses = ismissing(tab.Lifeexpectancy);
for i = length(misses):-1:1
    if misses(i) == 1
        tab(i,:) = [];
    end
end

missingIndices = ismissing(tab.AdultMortality);
tab(missingIndices, :) = [];

missingIndices = ismissing(tab.BMI);
tab(missingIndices, :) = [];

missingIndices = ismissing(tab.Polio);
tab(missingIndices, :) = [];

missingIndices = ismissing(tab.Diphtheria);
tab(missingIndices, :) = [];

missingIndices = ismissing(tab.("thinness1-19years"));
tab(missingIndices, :) = [];

missingIndices = ismissing(tab.("thinness5-9years"));
tab(missingIndices, :) = [];

missingIndices = ismissing(tab.HepatitisB);

X_train = tab.Diphtheria(~missingIndices);
y_train = tab.HepatitisB(~missingIndices);
X_test = tab.Diphtheria(missingIndices);

model = fitlm(X_train, y_train);
y_pred = predict(model, X_test);
tab.HepatitisB(missingIndices) = y_pred;

reg_test = predict(model, X_train);

err = mape(y_train, reg_test)

decisions = fuzzify(tab.Lifeexpectancy);
tab.Lifeexpectancy = [];

tab.Alcohol = fillAvg(tab.Alcohol, decisions);
tab.GDP = fillAvg(tab.GDP, decisions);
tab.Population = fillAvg(tab.Population, decisions);
tab.Totalexpenditure = fillAvg(tab.Totalexpenditure, decisions);
tab.Incomecompositionofresources = fillAvg(tab.Incomecompositionofresources, decisions);
tab.Schooling = fillAvg(tab.Schooling, decisions);

data = tab;
data = table2array(data(:,:));

means = zeros(1,size(data,2));
for i=1:size(data,2)
    means(i) = mean(data(:,i));
end

stds = zeros(1,size(data,2));
for i=1:size(data,2)
    stds(i) = std(data(:,i));
end

% normalize
for i=[1 3:13]
    data(:,i) = (data(:,i) - means(i)) / stds(i);
end

missing = any(ismissing(data(:,:)))
for i = 1:length(missing)
    if missing(i)
        data(:,i) = fillAvg(data(:,i), decisions);
    end
end
any(ismissing(data(:,:)))

corrMatrix = corr(data);

cors = triu(corrMatrix, 1);
rows = 1:(length(cors)-1)
delete = []
for row = rows
    for col = (row+1):length(cors)
        if abs(cors(row,col)) > 0.6
            delete = [delete, col];
        end
    end
end
delete = unique(delete);
delete = sort(delete, 'descend')
data(:, delete) = [];

vif_values = computeVIF(data) % usunąć powyżej 5-10

corrMatrix = corr(data);
heatmap(corrMatrix, 'Colormap', jet, 'ColorLimits', [-1, 1])

net = patternnet([24, 12, 6], 'trainlm');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';

net.trainParam.max_fail = 10;
net.trainParam.mu_dec = 0.015;
net.trainParam.mu_inc = 10;

[trainInd, valInd, testInd] = dividerand(size(data, 1), 0.70, 0.15, 0.15);

trainInputs = data(trainInd, :)';
trainTargets = decisions(:, trainInd);
valInputs = data(valInd, :)';
valTargets = decisions(:, valInd); 
testInputs = data(testInd, :)';
testTargets = decisions(:, testInd);

[net, tr] = train(net, trainInputs, trainTargets);

outputs = net(testInputs);
performance = perform(net, testTargets, outputs);
disp(['Performance na zbiorze testowym: ', num2str(performance)]);

% 35 + 45 /2 ?
ranges = [35 45 55 65 75 85]';
decodedOut = sum(outputs .* ranges);
decodedTargets = sum(testTargets .* ranges);

errors = decodedTargets - decodedOut;
rel_err = errors ./ decodedTargets;

figure;
scatter(1:length(decodedTargets), rel_err, 'filled');
xlabel('Indeks próbki');
ylabel('Błąd względny');
title('Wykres błędów względnych');
grid on;

figure;
scatter(1:length(decodedTargets), errors, 'filled');
xlabel('Indeks próbki');
ylabel('Błąd bezwzględny');
title('Wykres błędów bezwzględnych');
grid on;

mse1 = mse(errors)
mae1 = mae(errors)
mape1 = mape(decodedTargets, decodedOut)

figure;
scatter(decodedTargets, decodedOut, 'filled');
hold on;

plot([min(decodedTargets), max(decodedTargets)], [min(decodedTargets), max(decodedTargets)], 'r--', 'LineWidth', 2);

xlabel('Wartości rzeczywiste');
ylabel('Wartości przewidywane');
title('Wykres rozrzutu wyników przewidywanych i rzeczywistych');
grid on;
hold off;

%year status AdultMortality infantdeaths Alcohol percentageexpenditure
%HepatitisB  Measles BMI Polio Totalexpenditure HIV/AIDS thinness1-19years Incomecompositionofresources
exam = [15 1 263 62	 0.01 71 65	1154 19.1 6	 8.16 0.1 17.2 0.479; % 65
        30 1 263 62	 0.01 71 65	1154 19.1 6	 8.16 0.1 17.2 0.479; % +/-
        15 0 600 62	 0.01 71 65	1154 19.1 6  8.16 0.1 17.2 0.479; % -
        15 1 263 200 0.01 71 65 1154 19.1 6  8.16 0.1 17.2 0.479; % -
        15 1 263 62	 7.00 71 65 1154 19.1 6  8.16 0.1 17.2 0.479; % -
        15 1 263 62	 0.01 90 65	1154 19.1 6	 8.16 0.1 17.2 0.479; % +
        15 1 263 62	 0.01 71 99	1154 19.1 6	 8.16 0.1 17.2 0.479; % +
        15 1 263 62	 0.01 71 65	6500 19.1 6	 8.16 0.1 17.2 0.479; % -
        15 1 263 62	 0.01 71 65	1154 30.0 6	 8.16 0.1 17.2 0.479; % +
        15 1 263 62	 0.01 71 65	1154 19.1 99 8.16 0.1 17.2 0.479; % +
        15 1 263 62	 0.01 71 65	1154 19.1 6	 20.0 0.1 17.2 0.479; % +
        15 1 263 62	 0.01 71 65	1154 19.1 6	 8.16 25  17.2 0.479; % -
        15 1 263 62	 0.01 71 65	1154 19.1 6	 8.16 0.1 1.00 0.479; % +
        15 1 263 62	 0.01 71 65	1154 19.1 6	 8.16 0.1 17.2 0.800; % +
        20 1 130 30	 0.7  80 90	254  24.0 76 12.1 0.1 5.20 0.600;]';  % +++

for i=[1 3:13]
    exam(:,i) = (exam(:,i) - means(i)) / stds(i);
end

outputs2 = net(exam);
decodedOut2 = sum(outputs2 .* ranges);

C = confusionmat(vec2ind(testTargets), vec2ind(outputs));

figure;
heatmap(C, 'ColorBarVisible', 'off', 'XLabel', 'Przewidywana', 'YLabel', 'Rzeczywista', 'Title', 'Macierz Pomyłek');

num_classes = size(C, 1);

accuracy = zeros(num_classes, 1);
precision = zeros(num_classes, 1);
recall = zeros(num_classes, 1);
specificity = zeros(num_classes, 1);

for i = 1:num_classes
    TP = C(i, i);
    FP = sum(C(:, i)) - TP;
    FN = sum(C(i, :)) - TP;
    TN = sum(C(:)) - TP - FP - FN;
    
    accuracy(i) = (TP + TN) / sum(C(:));
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    specificity(i) = TN / (TN + FP);
end

for i = 1:num_classes
    disp(['Klasa ', num2str(i)]);
    disp(['Accuracy: ', num2str(accuracy(i))]);
    disp(['Recall: ', num2str(recall(i))]);
    disp(['Precision: ', num2str(precision(i))]);
    disp(['Specificity: ', num2str(specificity(i))]);
end

TP = sum(diag(C))
FN = sum(sum(triu(C, 1)));
FP = sum(sum(tril(C, -1)));
TN = sum(C(:)) - TP - FN - FP

accuracy = (TP + TN) / sum(C(:));
recall = TP / (TP + FN);
precision = TP / (TP + FP);

disp(['Accuracy: ', num2str(accuracy)]);
disp(['Recall: ', num2str(recall)]);
disp(['Precision: ', num2str(precision)]);


function vif = computeVIF(X)
    [n, p] = size(X);
    vif = zeros(1, p);
    for i = 1:p
        Xi = X(:, i);
        X_rest = X(:, [1:i-1, i+1:p]);
        lm = fitlm(X_rest, Xi);
        R2 = lm.Rsquared.Ordinary;
        vif(i) = 1 / (1 - R2);
    end
end

function coded = fuzzify(col)
    minm = min(col)-10;
    maxm = max(col);

    nrOfRanges = floor((maxm - minm)/10);
    coded = zeros(nrOfRanges, length(col));

    minm = floor(minm/10) * 10 + 5;    
    for nr = 1:nrOfRanges
        minm = minm+10;
        for i = 1:size(col)
            coded(nr, i) = max(1 - abs(0.1*(col(i)-minm)), 0);
        end
    end
end

function column = fillAvg(X, decisions)
    nrOfClasses = size(decisions, 1);
    
    avg = 0;
    nr = 0;
    for i = 1:size(X)
        if ~isnan(X(i))
            for cl = 1:nrOfClasses
                avg = avg + decisions(cl, i)*X(i);
                nr = nr+1;
            end
        end
    end
    avg = avg/nr;

    for i = 1:size(X)
        if isnan(X(i))
            X(i) = avg;
        end
    end
    column = X;
end