%% Objective Normalized Function Definition
function y = objective_norm(x)
    global Q_norm f_norm c_norm
    y = 0.5 * x' * Q_norm * x + f_norm' * x + c_norm; 
end