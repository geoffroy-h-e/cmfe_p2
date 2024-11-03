%% Load the data
global Q f c Q_norm f_norm c_norm
% Daily returns 
xlsx_daily_data = 'daily_data.xlsx';
daily_data = readtable(xlsx_daily_data, 'Range', 'A1:Z1395');

opts = detectImportOptions(xlsx_daily_data);
daily_data = readtable(xlsx_daily_data, opts);
disp(height(daily_data));  % Check if it reads 1395 rows

% daily indices
xlsx_daily_indices = 'daily_indices.xlsx';
daily_indices = readtable(xlsx_daily_indices);
log_daily_indices = log(daily_indices{1:end, 2:end} + 1);

% Time frames to iterate over (1 year, 2 years, 3 years)
time_frames = [365, 730, 1095];
years = ["Year 1", "Year 2", "Year 3"];

% Initialize tables to store results
results_table = table;        % Non-normalized weights
results_table_norm = table;    % Normalized weights


%% set up the problem 
for i = 1:length(time_frames)
    % Set n for the current iteration
    n = time_frames(i);

    % Select the data for the current time frame
    dates = daily_data.Caldt(1:n);
    r = daily_data.Dret(1:n);
    r = log(r + 1);
    r_norm = (r - mean(r)) ./ std(r);

    % Remove dates and combine in a single matrix
    F = log_daily_indices(1:n, 1:end);
    F_norm = (F - mean(F)) ./ std(F);

    % Ensure returns align 
    T = length(r);  % Number of time periods
    m = size(F, 2);  % Number of factors


    %% Problem constraints
    Aeq = ones(1, m);  % Equality constraint: sum(w) = 1
    beq = 1;
    A = -eye(m);  % Inequality constraints: w >= 0 
    b = zeros(m, 1);


    %% Optimization using quadprog
    ones_T = ones(T, 1);

    % Quadratic term (Q)
    Q = (2 / T) * (F' * F) - (2 / T^2) * (F' * ones_T) * (ones_T' * F);
    Q = (Q' + Q) / 2;

    % Quadratic term with normalized values (Q_norm)
    Q_norm = (2 / T) * (F_norm' * F_norm) - (2 / T^2) * (F_norm' * ones_T) * (ones_T' * F_norm);
    Q_norm = (Q_norm' + Q_norm) / 2;

    % Linear term (f)
    f = -(2 / T) * (F' * r) + (2 / T^2) * (ones_T' * r) * (F' * ones_T);

    % Linear term with normalized values (f_norm)
    f_norm = -(2 / T) * (F_norm' * r) + (2 / T^2) * (ones_T' * r) * (F_norm' * ones_T);

    % Constant term (c)
    c = (1 / T) * (r' * r) - (1 / T^2) * (ones_T' * r)^2;

    % Constant term with normalized values (c_norm)
    c_norm = (1 / T) * (r_norm' * r_norm) - (1 / T^2) * (ones_T' * r_norm)^2;

    % Constraints
    lb = zeros(m, 1);
    ub = ones(m, 1);

    % Solving with quadprog
    options_quadprog = optimoptions('quadprog', 'Display', 'off');
    [w_opt_quadprog, ~] = quadprog(Q, f, [], [], Aeq, beq, lb, ub, [], options_quadprog);

    % Solving with quadprog with the normalized Q and f
    [w_opt_quadprog_norm, ~] = quadprog(Q_norm, f_norm, [], [], Aeq, beq, lb, ub, [], options_quadprog);


    %% Optimization using fmincon
    % Initial guess for weights
    w0 = ones(m, 1) / m;

    % Solving with fmincon
    options_fmincon = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off', 'MaxIterations', 1000, 'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-9);
    [w_opt_fmincon, ~] = fmincon(@objective, w0, [], [], Aeq, beq, lb, ub, [], options_fmincon);

    % Solving with fmincon with the normalized values
    [w_opt_fmincon_norm, ~] = fmincon(@objective_norm, w0, [], [], Aeq, beq, lb, ub, [], options_fmincon);


    %% Store the Results in Separate Tables
    % Create a temporary table for non-normalized weights for the current year
    temp_table = table(w_opt_quadprog, w_opt_fmincon, ...
                       'VariableNames', {['Quadprog_', char(years(i))], ['Fmincon_', char(years(i))]});
    % Append to the non-normalized results table
    results_table = [results_table, temp_table];

    % Create a temporary table for normalized weights for the current year
    temp_table_norm = table(w_opt_quadprog_norm, w_opt_fmincon_norm, ...
                            'VariableNames', {['Quadprog_Norm_', char(years(i))], ['Fmincon_Norm_', char(years(i))]});
    % Append to the normalized results table
    results_table_norm = [results_table_norm, temp_table_norm];


    %% Calculate and Display Additional Metrics
    % Non-normalized
    uP_quadprog = r - F * w_opt_quadprog;  % Residuals for quadprog
    uP_fmincon = r - F * w_opt_fmincon;    % Residuals for fmincon
    diff_non_norm = norm(w_opt_quadprog - w_opt_fmincon);
    excess_return_quadprog = sum(uP_quadprog);
    excess_return_fmincon = sum(uP_fmincon);

    % Display non-normalized metrics
    disp(['Metrics for ', char(years(i)), ' (Non-Normalized):']);
    disp(['  Norm of Difference: ', num2str(diff_non_norm)]);
    disp(['  Excess Return (Quadprog): ', num2str(excess_return_quadprog), '   Variance: ', num2str(objective(w_opt_quadprog))]);

    disp(['  Excess Return (Fmincon): ', num2str(excess_return_fmincon), '   Variance: ', num2str(objective(w_opt_fmincon))]);

    % Normalized
    uP_quadprog_norm = r_norm - F * w_opt_quadprog_norm;  % Residuals for quadprog
    uP_fmincon_norm = r_norm - F_norm * w_opt_fmincon_norm;    % Residuals for fmincon
    diff_norm = norm(w_opt_quadprog_norm - w_opt_fmincon_norm);
    excess_return_quadprog_norm = sum(uP_quadprog_norm);
    excess_return_fmincon_norm = sum(uP_fmincon_norm);

    % Display normalized metrics
    disp(['Metrics for ', char(years(i)), ' (Normalized):']);
    disp(['  Norm of Difference: ', num2str(diff_norm)]);
    disp(['  Excess Return (Quadprog): ', num2str(excess_return_quadprog_norm), '   Variance: ', num2str(objective_norm(w_opt_quadprog_norm))]);
    disp(['  Excess Return (Fmincon): ', num2str(excess_return_fmincon_norm), '   Variance: ', num2str(objective_norm(w_opt_fmincon_norm))]);
end

%% Display the results tables
disp('Optimal Weights for each year (Non-Normalized):');
disp(results_table);

disp('Optimal Weights for each year (Normalized):');
disp(results_table_norm);


%% Plotting Weights by Year and Optimization Method (Grouped by quadprog and fmincon)

% Define factor labels
factor_labels = {
    'S&P U.S. Treasury Bill Index', 'S&P 400 (mid cap)', 'S&P 600 (small cap)', ...
    'S&P 500 (large cap)', 'S&P 500 Growth', 'S&P 500 Value', ...
    'S&P 500 Information Technology Index', 'Technology Select Sector Index'};

% Iterate over each year for both quadprog and fmincon
for i = 1:3
    % Extract non-normalized weights for quadprog and fmincon for the current year
    weights_quadprog_non_normalized = results_table{:, (i-1)*2 + 1};      % Quadprog non-normalized
    weights_fmincon_non_normalized = results_table{:, (i-1)*2 + 2};       % Fmincon non-normalized
    weights_quadprog_normalized = results_table_norm{:, (i-1)*2 + 1};     % Quadprog normalized
    weights_fmincon_normalized = results_table_norm{:, (i-1)*2 + 2};      % Fmincon normalized

    % Plot non-normalized weights for the current year (Quadprog vs Fmincon)
    figure;
    subplot(1, 2, 1);
    bar(weights_quadprog_non_normalized, 'FaceColor', [0.2, 0.6, 0.8]);
    title(['Optimal Weights - ', years(i), ' (Non-Normalized, Quadprog)']);
    set(gca, 'XTickLabel', factor_labels);
    ylabel('Weight');
    xtickangle(45);

    subplot(1, 2, 2);
    bar(weights_fmincon_non_normalized, 'FaceColor', [0.2, 0.8, 0.6]);
    title(['Optimal Weights - ', years(i), ' (Non-Normalized, Fmincon)']);
    set(gca, 'XTickLabel', factor_labels);
    ylabel('Weight');
    xtickangle(45);

    % Save figure
    saveas(gcf, ['Year_', num2str(i), '_NonNormalized.png']);

    % Plot normalized weights for the current year (Quadprog vs Fmincon)
    figure;
    subplot(1, 2, 1);
    bar(weights_quadprog_normalized, 'FaceColor', [0.8, 0.4, 0.2]);
    title(['Optimal Weights - ', years(i), ' (Normalized, Quadprog)']);
    set(gca, 'XTickLabel', factor_labels);
    ylabel('Weight');
    xtickangle(45);

    subplot(1, 2, 2);
    bar(weights_fmincon_normalized, 'FaceColor', [0.8, 0.6, 0.2]);
    title(['Optimal Weights - ', years(i), ' (Normalized, Fmincon)']);
    set(gca, 'XTickLabel', factor_labels);
    ylabel('Weight');
    xtickangle(45);

    % Save figure
    saveas(gcf, ['Year_', num2str(i), '_Normalized.png']);
end