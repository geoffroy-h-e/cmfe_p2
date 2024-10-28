
%% load the data

% daily_returns 
xlsx_daily_data = 'daily_data.xlsx';
daily_data = readtable(xlsx_daily_data); 

%fama and french factors
xlsx_daily_fama5_factors = 'daily_fama5_factors.xlsx';
daily_fama5_factors = readtable(xlsx_daily_fama5_factors); 

% hou xue q5 factors model
xlsx_daily_q5_factors = 'daily_q5_factors.csv';
daily_q5_factors = readtable(xlsx_daily_q5_factors);

%set up the number of observations to test different time frames 
n = 1000;
dates = daily_data.Caldt(1:n);
r = daily_data.Dret(1:n);

%remove dates and combine in a single matrix
Fama5 = daily_fama5_factors{1:n, 2:end};  % Fama-French 5-factors
Q5 = daily_q5_factors{1:n, 2:end};   % Q5-factors
F = [Fama5, Q5];

% Ensure returns align 
T = length(r);  % Number of time periods
m = size(F, 2);  % Number of factors

%% correlation matrix to check if some factors have an alarming correlation 

corr_matrix = corr(F);
disp('Correlation Matrix of Factors:');
disp(corr_matrix);

%% setting up the optimization problem 

% Hessian matrix
I = eye(T);
ones_T = ones(T, 1);
H = (1/T) * (F' * (I - (1/T) * (ones_T * ones_T')) * F);

% Linear term
f = -2 * ((r' * F) / T - (sum(r) / T^2) * (ones_T' * F));

% Constraints: sum(w) = 1, w >= 0
Aeq = ones(1, m);  % Equality constraint: sum(w) = 1
beq = 1;
A = -eye(m);  % Inequality constraints: w over or equal to 0 
b = zeros(m, 1);

% Run the quadratic programming solver, 
% added tweak on tolerance to avoid getting stuck with a 1 weight 
options = optimoptions('quadprog', 'Display', 'off', 'TolFun', 1e-9, 'TolX', 1e-9);
[w_opt, ~] = quadprog(H, f', A, b, Aeq, beq, [], [], [], options);
%[w_opt, ~] = quadprog(H, f', A, b, Aeq, beq, lb, ub, x0, options);

% selection effect (u_P)
uP = r - F * w_opt;  % Residuals

w_opt

