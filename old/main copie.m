
%% Load the data

% Daily returns 
xlsx_daily_data = 'daily_data.xlsx';
daily_data = readtable(xlsx_daily_data); 

% daily indices
xlsx_daily_indices = 'daily_indices.xlsx';
daily_indices = readtable(xlsx_daily_indices);

% Fama and French factors
xlsx_daily_fama5_factors = 'daily_fama5_factors.xlsx';
daily_fama5_factors = readtable(xlsx_daily_fama5_factors); 

% Hou, Xue, and Zhang Q5 factors
xlsx_daily_q5_factors = 'daily_q5_factors.csv';
daily_q5_factors = readtable(xlsx_daily_q5_factors);

% Set up the number of observations to test different time frames 
n = 850;
dates = daily_data.Caldt(1:n);
r = daily_data.Dret(1:n);

% Remove dates and combine in a single matrix
Fama5 = daily_fama5_factors{1:n, 2:end};  % Fama-French 5-factors
Q5 = daily_q5_factors{1:n, 2:end}; % Q5-factors
F = daily_indices{1:n, 2:end};
%F = [Fama5, Q5];

% Ensure returns align 
T = length(r);  % Number of time periods
m = size(F, 2);  % Number of factors

%% Correlation matrix to check for alarming correlations 

corr_matrix = corr(F);
disp('Correlation Matrix of Factors:');
disp(corr_matrix);

%% Problem constraints

Aeq = ones(1, m);  % Equality constraint: sum(w) = 1
beq = 1;
A = -eye(m);  % Inequality constraints: w >= 0 
b = zeros(m, 1);

%% Optimization using quadprog

% Hessian matrix
I = eye(T);
ones_T = ones(T, 1);
H = (1/T) * (F' * (I - (1/T) * (ones_T * ones_T')) * F);

% Linear term
f = -2 * ((r' * F) / T - (sum(r) / T^2) * (ones_T' * F));

% solving with quadprod, tolerance adjusted to avoid getting stuck
options_quadprog = optimoptions('quadprog', 'Display', 'off', 'TolFun', 1e-9, 'TolX', 1e-9);
[w_opt_quadprog, ~] = quadprog(H, f', A, b, Aeq, beq, [], [], [], options_quadprog);

%% Optimization using using fmicon

% Objective function for fmincon
objective = @(w) (1/T) * norm(r - F * w)^2 - (1/T^2) * (sum(r - F * w))^2;

% Initial guess for weights
w0 = ones(m, 1) / m;

% Solving with fmincon
options_fmincon = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'off');
[w_opt_fmincon, ~] = fmincon(objective, w0, A, b, Aeq, beq, [], [], [], options_fmincon);

%% Compare the Results

disp('Optimal Weights using quadprog:');
disp(w_opt_quadprog);

disp('Optimal Weights using fmincon:');
disp(w_opt_fmincon);

%% Selection effect u(P)

uP_quadprog = r - F * w_opt_quadprog;  % Residuals for quadprog
uP_fmincon = r - F * w_opt_fmincon;    % Residuals for fmincon

%% Check diff between Solutions

difference = norm(w_opt_quadprog - w_opt_fmincon);
disp(['Norm of Difference between Solutions: ', num2str(difference)]);
