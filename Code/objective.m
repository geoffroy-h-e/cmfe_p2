%% Objective Function Definition
function y = objective(x)
    global Q f c
    y = 0.5 * x' * Q * x + f' * x + c; 
end
