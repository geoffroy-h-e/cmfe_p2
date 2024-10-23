
% load in the data
xlsx_daily_data = 'daily_data.xlsx';
daily_data = readtable(xlsx_daily_data);
xlsx_daily_factors = 'daily_factors.xlsx';
daily_factors = readtable(xlsx_daily_factors);

dates = daily_data.Caldt;
r = daily_data.Dret;  % Mutual funds returns
F = daily_factors{:, 2:end};  % Benchmark/factor returns (T x m)

% Ensure returns align by date, trim if needed
T = length(r);  % Number of time periods
m = size(F, 2);  % Number of factors

% Calculate intermediate terms
r_norm_sq = norm(r)^2;  % ||r||^2
r_mean_sum_sq = (sum(r))^2;  % (1^T r)^2

% Hessian matrix
I = eye(T);
ones_T = ones(T, 1);
H = (1/T) * (F' * (I - (1/T) * (ones_T * ones_T')) * F);

% Linear term
f = -2 * ((r' * F) / T - (sum(r) / T^2) * (ones_T' * F));

% Constraints: sum(w) = 1, w >= 0
Aeq = ones(1, m);  % Equality constraint: sum(w) = 1
beq = 1;
A = -eye(m);  % Inequality constraints: w >= 0
b = zeros(m, 1);

%lb = zeros(m, 1);  % Lower bounds: 0
%ub = ones(m, 1) * 0.3; test with upper bounds to see if we are stuck 

% Run the quadratic programming solver
options = optimoptions('quadprog', 'Display', 'off', 'TolFun', 1e-9, 'TolX', 1e-9);
[w_opt, ~] = quadprog(H, f', A, b, Aeq, beq, [], [], [], options);
%[w_opt, ~] = quadprog(H, f', A, b, Aeq, beq, lb, ub, x0, options);

% selection effect (u_P)
uP = r - F * w_opt;  % Residuals

w_opt

corr_matrix = corr(F);
disp('Correlation Matrix of Factors:');
disp(corr_matrix);
